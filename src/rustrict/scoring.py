"""Unified weighted scoring engine combining HTML, ML, semantic, DGA, IP/ASN, MX, and RDAP signals."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from .classifier import (
    ContentClassifier,
    DomainClassifier,
    html_keyword_hits,
    match_hard_html_junk,
)
from .dns_geo import DnsResult
from .filters import SKETCHY_TLDS, TaintIndex, dga_signals, has_mirror_shape
from .normalize import PSL, DomainMatcher
from .probe import PageProbe
from .reputation import IpReputation
from .semantic import SemanticClassifier, SemanticScore

log = logging.getLogger(__name__)

DROP_THRESHOLD = 100
RDAP_BORDERLINE_LO = 35
RDAP_BORDERLINE_HI = 135


@dataclass
class ScoreResult:
    domain: str
    score: int
    dropped: bool
    primary_reason: str
    reasons: dict[str, int] = field(default_factory=dict)


def _add(reasons: dict[str, int], key: str, pts: int) -> None:
    if pts != 0:
        reasons[key] = reasons.get(key, 0) + pts


def _apply_domain_age_points(reasons: dict[str, int], age_days: int | None) -> None:
    if age_days is None:
        return
    if age_days < 90:
        _add(reasons, "rdap:new_lt90d", 45)
    elif age_days < 180:
        _add(reasons, "rdap:new_lt180d", 25)
    elif age_days > 5 * 365:
        _add(reasons, "rdap:old_gt5y", -40)
    elif age_days > 3 * 365:
        _add(reasons, "rdap:old_gt3y", -25)


def _apply_crux_points(
    reasons: dict[str, int], rank: int | None, sketchy_tld: bool
) -> None:
    if rank is None:
        return
    if rank <= 10_000:
        _add(reasons, "crux:top10k", -40)
    elif rank <= 50_000:
        _add(reasons, "crux:top50k", -20)
    elif sketchy_tld and rank > 100_000:
        _add(reasons, "crux:tail_sketchy_tld", 25)
    elif rank > 500_000:
        _add(reasons, "crux:tail_500k", 15)


def _apply_semantic_points(reasons: dict[str, int], sem: SemanticScore | None) -> None:
    if sem is None:
        return
    m = sem.margin
    if m >= 0.08:
        _add(reasons, f"sem:{sem.junk_category}", 85)
    elif m >= 0.05:
        _add(reasons, f"sem:{sem.junk_category}", 55)
    elif m >= 0.025:
        _add(reasons, f"sem:{sem.junk_category}", 25)
    elif m <= -0.06:
        _add(reasons, f"sem:legit_{sem.legit_category}", -45)
    elif m <= -0.025:
        _add(reasons, f"sem:legit_{sem.legit_category}", -20)


def _apply_ml_points(
    reasons: dict[str, int], domain_prob: float, content_prob: float | None
) -> None:
    if domain_prob >= 0.92:
        _add(reasons, "ml:domain_high", 75)
    elif domain_prob >= 0.80:
        _add(reasons, "ml:domain_med", 45)
    elif domain_prob >= 0.60:
        _add(reasons, "ml:domain_low", 20)

    if content_prob is not None:
        if content_prob >= 0.85:
            _add(reasons, "ml:content_high", 70)
        elif content_prob >= 0.65:
            _add(reasons, "ml:content_med", 35)
        elif content_prob <= 0.15:
            _add(reasons, "ml:content_legit", -20)


def _primary_reason(reasons: dict[str, int]) -> str:
    if not reasons:
        return "keep"
    return max(reasons.items(), key=lambda kv: kv[1])[0]


def score_and_filter_domains(
    domains: list[str],
    probes: dict[str, PageProbe],
    dns_results: dict[str, DnsResult],
    mx_domains: set[str],
    crux_ranks: dict[str, int],
    domain_clf: DomainClassifier,
    semantic_clf: SemanticClassifier | None,
    taint: TaintIndex,
    ip_rep: IpReputation,
    psl: PSL,
    feed_matcher: DomainMatcher,
    allow_matcher: DomainMatcher,
    legit_seed_probes: list[PageProbe] | None = None,
    rdap_age_fn: Callable[[list[str]], dict[str, int]] | None = None,
) -> tuple[list[str], list[ScoreResult]]:
    """Evaluate all candidate domains via hard rules + unified weighted scoring."""
    domain_probs = dict(zip(domains, domain_clf.predict_scores(domains)))

    # 1. Semantic zero-shot embeddings over all fetched HTML pages.
    semantic_scores: dict[str, SemanticScore] = {}
    if semantic_clf is not None and probes:
        semantic_scores = semantic_clf.score_probes(
            probes[d] for d in domains if d in probes
        )

    # 2. Train HTML+Domain distant-supervision ContentClassifier from clear seeds.
    junk_html_seeds: list[PageProbe] = []
    legit_html_seeds: list[PageProbe] = list(legit_seed_probes or [])
    html_candidates: list[PageProbe] = []

    for d in domains:
        probe = probes.get(d)
        if not probe or probe.status != "ok" or not probe.combined_text():
            continue
        html_candidates.append(probe)
        hard = match_hard_html_junk(probe, psl, feed_matcher)
        _, kw_hits = html_keyword_hits(probe)
        sem = semantic_scores.get(d)
        if (
            hard
            or kw_hits >= 2
            or domain_probs[d] >= 0.92
            or (sem and sem.margin >= 0.08)
        ):
            junk_html_seeds.append(probe)
        elif (
            domain_probs[d] <= 0.15
            and kw_hits == 0
            and (not sem or sem.margin <= -0.02)
        ):
            legit_html_seeds.append(probe)

    content_clf = ContentClassifier()
    content_clf.fit(junk_html_seeds, legit_html_seeds)
    content_probs: dict[str, float] = {}
    if content_clf.fitted and html_candidates:
        content_probs = dict(
            zip(
                (p.domain for p in html_candidates),
                content_clf.predict_scores(html_candidates),
            )
        )

    # 3. Pre-score every domain before selective RDAP lookups.
    pre_reasons: dict[str, dict[str, int]] = {}
    hard_decisions: dict[str, ScoreResult] = {}

    for d in domains:
        if allow_matcher.match(d) or psl.is_public_suffix(d):
            hard_decisions[d] = ScoreResult(d, -100, False, "allowlist")
            continue

        probe = probes.get(d)
        if probe is not None:
            hard = match_hard_html_junk(probe, psl, feed_matcher)
            if hard:
                hard_decisions[d] = ScoreResult(d, 200, True, hard, {hard: 200})
                continue
            if probe.status == "dead":
                reason = "mail_only_spam" if d in mx_domains else "probe:dead"
                hard_decisions[d] = ScoreResult(d, 200, True, reason, {reason: 200})
                continue

        reasons: dict[str, int] = {}
        tld = d.rsplit(".", 1)[-1]
        sketchy_tld = tld in SKETCHY_TLDS
        mirror_shape = has_mirror_shape(d, psl)
        dga = dga_signals(d, psl)

        # Structural / DGA / TLD / tainting signals
        if sketchy_tld:
            _add(reasons, "struct:sketchy_tld", 25)
        if dga["leetspeak"]:
            _add(reasons, "dga:leetspeak", 55)
        if dga["consonant_cluster"] or dga["low_vowels"]:
            _add(reasons, "dga:random_chars", 55)

        if taint.root_is_tainted(d, psl):
            _add(reasons, "taint:root", 75)
        elif taint.stem_drop_count(d, psl) >= 2:
            _add(reasons, "taint:stem", 60 if mirror_shape else 35)

        # CrUX popularity adjustment
        _apply_crux_points(reasons, crux_ranks.get(d), sketchy_tld)

        # HTML keyword & content signals
        has_usable_html = bool(
            probe and probe.status == "ok" and len(probe.combined_text()) >= 15
        )
        if probe and probe.status == "ok":
            kw_cat, kw_hits = html_keyword_hits(probe)
            if kw_cat and kw_hits >= 2:
                _add(reasons, f"html:{kw_cat}", 95)
            elif kw_cat and kw_hits == 1:
                _add(reasons, f"html_weak:{kw_cat}", 30)

        # Grey-zone default-deny: suspicious domain with no readable HTML (403/JS-SPA/non-HTML)
        if (
            probe is not None
            and not has_usable_html
            and (
                sketchy_tld
                or dga["leetspeak"]
                or dga["consonant_cluster"]
                or taint.stem_drop_count(d, psl) >= 2
            )
        ):
            _add(reasons, "grey:no_html_suspicious", 35)
        if probe is not None and probe.status == "non_html" and d in mx_domains:
            _add(reasons, "mx:non_html_mail", 60)

        # ML & Semantic zero-shot signals
        _apply_ml_points(reasons, domain_probs[d], content_probs.get(d))
        _apply_semantic_points(reasons, semantic_scores.get(d))

        # IP & ASN reputation
        dns_res = dns_results.get(d)
        if dns_res:
            rep = ip_rep.evaluate(dns_res.ips)
            if rep["bad_ip"]:
                _add(reasons, "rep:bad_ip", 80)
            if rep["bad_asn"]:
                _add(reasons, "rep:bad_asn", 80)
            if rep["cohosted_junk_ip"]:
                _add(reasons, "rep:cohosted_junk_ip", 60)

        pre_reasons[d] = reasons

    # 4. Query RDAP domain age ONLY for borderline domains to break ties accurately.
    rdap_ages: dict[str, int] = {}
    if rdap_age_fn is not None:
        borderline_roots = [
            psl.registrable(d)
            for d, r in pre_reasons.items()
            if RDAP_BORDERLINE_LO <= sum(r.values()) <= RDAP_BORDERLINE_HI
        ]
        if borderline_roots:
            rdap_ages = rdap_age_fn(borderline_roots)

    # 5. Finalize scores and decisions.
    kept: list[str] = []
    all_results: list[ScoreResult] = []
    for d in domains:
        if d in hard_decisions:
            res = hard_decisions[d]
            all_results.append(res)
            if not res.dropped:
                kept.append(d)
            continue

        reasons = pre_reasons[d]
        _apply_domain_age_points(reasons, rdap_ages.get(psl.registrable(d)))
        total = sum(reasons.values())
        dropped = total >= DROP_THRESHOLD
        res = ScoreResult(
            domain=d,
            score=total,
            dropped=dropped,
            primary_reason=_primary_reason(reasons),
            reasons=reasons,
        )
        all_results.append(res)
        if not dropped:
            kept.append(d)

    return kept, all_results
