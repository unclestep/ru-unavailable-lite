"""CLI entrypoint for building rustrict release artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .classifier import DomainClassifier
from .crux import domain_crux_ranks, fetch_crux_snapshot, load_crux_snapshot, save_crux_snapshot
from .dns_geo import CountryDB, check_mx_records, is_ru_hosted, resolve_domains
from .export import build_geoip_lists, decode_geoip, write_artifacts
from .filters import TaintIndex, apply_deterministic_filters, build_stem_counts
from .normalize import (
    DomainMatcher,
    PSL,
    dedup_by_coverage,
    filter_subdomain_farms,
    load_domain_file,
    normalize_domain,
)
from .probe import PageProbe, probe_domains
from .reputation import fetch_domain_ages_days, fetch_ip_reputation
from .scoring import ScoreResult, score_and_filter_domains
from .semantic import SemanticClassifier
from .sources import (
    FOREIGN_GEOIP_URL,
    GEOIP_IPV4_URL,
    GEOIP_IPV6_URL,
    PSL_URL,
    fetch_branch_sources,
    fetch_google_first_party_cidrs,
    fetch_junk_feeds,
    fetch_legit_feeds,
    make_session,
)

log = logging.getLogger("rustrict")


def _load_psl(work_dir: Path, session) -> PSL:
    psl_cache = work_dir / "psl.dat"
    if psl_cache.exists():
        return PSL(psl_cache.read_text(encoding="utf-8"))
    text = session.get(PSL_URL, timeout=120).text
    work_dir.mkdir(parents=True, exist_ok=True)
    psl_cache.write_text(text, encoding="utf-8")
    return PSL(text)


def stage_crux(project_id: str, work_dir: Path, rank_max: int) -> None:
    session = make_session()
    psl = _load_psl(work_dir, session)
    snap = fetch_crux_snapshot(project_id, psl, rank_max=rank_max)
    save_crux_snapshot(work_dir / "crux.json.gz", snap)


def _strip_secrets_from_env() -> None:
    """Drop tokens/credentials before probing arbitrary untrusted third-party sites."""
    for key in (
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "CLOUDSDK_AUTH_ACCESS_TOKEN",
    ):
        os.environ.pop(key, None)


def _write_audit(path: Path, results: list[ScoreResult]) -> None:
    kept_scored = sorted((r for r in results if not r.dropped and r.primary_reason != "allowlist"), key=lambda r: -r.score)
    dropped_scored = sorted((r for r in results if r.dropped and r.score < 200), key=lambda r: r.score)

    audit = {
        "dropped_by_reason": dict(Counter(r.primary_reason for r in results if r.dropped).most_common()),
        "borderline_kept_top60": [asdict(r) for r in kept_scored[:60]],
        "borderline_dropped_top60": [asdict(r) for r in dropped_scored[:60]],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stage_build(
    work_dir: Path,
    out_dir: Path,
    allowlist_path: Path,
    cache_path: Path | None,
    rdap_cache_path: Path | None,
    version: str,
    skip_probe: bool,
    skip_semantic: bool,
    skip_rdap: bool,
    probe_concurrency: int,
) -> None:
    stats: dict = {}

    auth_session = make_session(github_token=os.environ.get("GITHUB_TOKEN"))
    psl = _load_psl(work_dir, auth_session)
    crux = load_crux_snapshot(work_dir / "crux.json.gz")
    stats["crux_yyyymm"] = crux.yyyymm

    # 1. Fetch sources, threat feeds, GeoIP/ASN/IP-reputation DBs, and Google first-party CIDRs.
    raw = fetch_branch_sources(auth_session)
    junk_feeds = fetch_junk_feeds(auth_session)
    legit_ut1 = fetch_legit_feeds(auth_session)

    geo_v4_csv = auth_session.get(GEOIP_IPV4_URL, timeout=300).text
    geo_v6_csv = auth_session.get(GEOIP_IPV6_URL, timeout=300).text
    country_db = CountryDB(geo_v4_csv, geo_v6_csv)

    ip_rep = fetch_ip_reputation(auth_session)
    foreign_geoip = decode_geoip(auth_session.get(FOREIGN_GEOIP_URL, timeout=300).content)
    google_cidrs = fetch_google_first_party_cidrs(auth_session)

    # Close authenticated session and scrub tokens before any untrusted HTTP crawl.
    auth_session.close()
    _strip_secrets_from_env()

    # 2. Normalize and deduplicate both branches.
    allowlist = load_domain_file(allowlist_path)
    ru_forbidden = dedup_by_coverage(d for x in raw["ru-forbidden"] if (d := normalize_domain(x)))
    ru_blocked_all = dedup_by_coverage(d for x in raw["ru-blocked"] if (d := normalize_domain(x)))
    stats["raw_normalized"] = {"ru_blocked": len(ru_blocked_all), "ru_forbidden": len(ru_forbidden)}

    stem_counts = build_stem_counts(ru_blocked_all, psl)

    # 3. ru-forbidden: untouched community list except dropping stray RU-hosted entries.
    forb_dns = resolve_domains(ru_forbidden)
    forb_ru = [d for d in ru_forbidden if is_ru_hosted(forb_dns[d].ips, country_db)]
    ru_forbidden = [d for d in ru_forbidden if d not in set(forb_ru)]
    stats["ru_forbidden_dropped_ru_host"] = len(forb_ru)
    log.info("ru-forbidden: kept %d, dropped RU-hosted %d", len(ru_forbidden), len(forb_ru))

    allow_matcher = DomainMatcher(allowlist | set(ru_forbidden))

    # 4. ru-blocked: remove subdomain farms, then filter by CrUX RU popularity.
    ru_blocked, farms = filter_subdomain_farms(ru_blocked_all, psl, max_subdomains=5, protected=allow_matcher)
    stats["subdomain_farms_dropped"] = sum(farms.values())
    log.info("Subdomain farms dropped: %d domains across %d roots", sum(farms.values()), len(farms))

    crux_rank_map = domain_crux_ranks(ru_blocked, crux, psl)
    ru_blocked = list(crux_rank_map)
    stats["after_crux"] = len(ru_blocked)
    log.info("ru-blocked after CrUX: %d", len(ru_blocked))

    # 5. Single-pass DNS liveness + RU hosting removal for ru-blocked.
    blocked_dns = resolve_domains(ru_blocked)
    alive_blocked: list[str] = []
    dead_cnt = ru_host_cnt = 0
    for d in ru_blocked:
        res = blocked_dns[d]
        if res.status != "alive":
            dead_cnt += 1
            continue
        if is_ru_hosted(res.ips, country_db):
            ru_host_cnt += 1
            continue
        alive_blocked.append(d)
    ru_blocked = alive_blocked
    stats["dns_geo_dropped"] = {"dead": dead_cnt, "ru_hosted": ru_host_cnt}
    log.info("ru-blocked after DNS+GeoIP: %d (dead=%d, ru_hosted=%d)", len(ru_blocked), dead_cnt, ru_host_cnt)

    # 6. Deterministic filters: feeds + strict RU junk rules + mirror clusters.
    all_junk_feed = set().union(*junk_feeds.values()) if junk_feeds else set()
    feed_matcher = DomainMatcher(all_junk_feed)
    ru_blocked, det_dropped = apply_deterministic_filters(
        ru_blocked, psl, feed_matcher, allow_matcher, stem_counts
    )
    stats["deterministic_dropped"] = {k: len(v) for k, v in det_dropped.items()}
    log.info("ru-blocked after deterministic filters: %d (%s)", len(ru_blocked), stats["deterministic_dropped"])

    # Build similarity taint index & dedicated-IP co-hosting taint from already-dropped junk.
    all_det_dropped = [d for items in det_dropped.values() for d in items]
    taint = TaintIndex.build([*all_det_dropped, *farms.keys()], psl, allow_matcher)
    cohosted_ips = ip_rep.build_cohosting_taint(
        blocked_dns[d].ips for d in all_det_dropped if d in blocked_dns
    )
    stats["tainted_cohosted_ips"] = len(cohosted_ips)

    # 7. Train Domain+Structural ML classifier.
    rng = random.Random(42)
    train_junk_set: set[str] = set()
    for key in ("ut1/gambling", "ut1/drogue", "ut1/cryptojacking", "ut1/ddos", "ut1/stalkerware"):
        train_junk_set.update(junk_feeds.get(key, set()))
    for key in ("porn", "gambling_extra", "hagezi/tif"):
        items = sorted(junk_feeds.get(key, set()))
        train_junk_set.update(rng.sample(items, min(len(items), 25_000)))
    for key, items in det_dropped.items():
        if key != "feed":
            train_junk_set.update(items)

    porn_matcher = DomainMatcher(junk_feeds.get("porn", set()))
    clean_forbidden = [d for d in ru_forbidden if not porn_matcher.match(d)]
    train_legit_set = (legit_ut1 | allowlist | set(clean_forbidden)) - train_junk_set
    train_junk_set -= train_legit_set

    domain_clf = DomainClassifier(psl, stem_counts)
    stats["domain_ml_auc"] = round(domain_clf.fit(sorted(train_junk_set), sorted(train_legit_set)), 4)

    # 8. SSRF-safe HTML probe + MX check on web-dead/non-HTML domains.
    probes: dict[str, PageProbe] = {}
    legit_seed_probes: list[PageProbe] = []
    mx_domains: set[str] = set()
    if not skip_probe:
        seed_sample = rng.sample(clean_forbidden, min(len(clean_forbidden), 400))
        to_probe = sorted(set(ru_blocked) | set(seed_sample))
        probes = probe_domains(to_probe, cache_path=cache_path, concurrency=probe_concurrency)
        legit_seed_probes = [
            probes[d]
            for d in seed_sample
            if d in probes and probes[d].status == "ok" and probes[d].combined_text()
        ]

        no_web = [d for d in ru_blocked if d in probes and probes[d].status in ("dead", "non_html")]
        if no_web:
            mx_domains = check_mx_records(no_web)
            log.info("MX check on %d web-dead/non-HTML domains: %d have MX", len(no_web), len(mx_domains))

    # 9. Unified weighted scoring (HTML + Semantic Zero-Shot + ML + DGA + Taint + IP/ASN + MX + RDAP).
    semantic_clf = None if (skip_probe or skip_semantic) else SemanticClassifier()
    rdap_fn = (
        None
        if skip_rdap
        else (lambda roots: fetch_domain_ages_days(roots, cache_path=rdap_cache_path))
    )

    ru_blocked, score_results = score_and_filter_domains(
        ru_blocked,
        probes=probes,
        dns_results=blocked_dns,
        mx_domains=mx_domains,
        crux_ranks=crux_rank_map,
        domain_clf=domain_clf,
        semantic_clf=semantic_clf,
        taint=taint,
        ip_rep=ip_rep,
        psl=psl,
        feed_matcher=feed_matcher,
        allow_matcher=allow_matcher,
        legit_seed_probes=legit_seed_probes,
        rdap_age_fn=rdap_fn,
    )
    score_drop_counts = dict(Counter(r.primary_reason for r in score_results if r.dropped).most_common())
    stats["scoring_dropped"] = score_drop_counts
    log.info("ru-blocked after unified scoring: %d (%s)", len(ru_blocked), score_drop_counts)

    _write_audit(work_dir / "audit.json", score_results)

    # 10. Final deduplication and export (ru-forbidden takes priority over ru-blocked).
    ru_forbidden = dedup_by_coverage(ru_forbidden)
    forb_matcher = DomainMatcher(ru_forbidden)
    ru_blocked = [d for d in dedup_by_coverage(ru_blocked) if not forb_matcher.match(d)]

    geoip_lists = build_geoip_lists(foreign_geoip, google_cidrs, [*ru_blocked, *ru_forbidden], psl)
    write_artifacts(out_dir, version, ru_blocked, ru_forbidden, geoip_lists, stats)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build rustrict routing lists")
    parser.add_argument("--stage", choices=("crux", "build", "all"), default="all")
    parser.add_argument("--project-id", default=os.environ.get("GOOGLE_CLOUD_PROJECT", "rkn-block-filtering"))
    parser.add_argument("--rank-max", type=int, default=1_000_000)
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--out-dir", type=Path, default=Path("publish"))
    parser.add_argument("--allowlist", type=Path, default=Path("allowlist.txt"))
    parser.add_argument("--cache-file", type=Path, default=Path("work/probe-cache.json.gz"))
    parser.add_argument("--rdap-cache", type=Path, default=Path("work/rdap-cache.json.gz"))
    parser.add_argument("--version", default=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M"))
    parser.add_argument("--skip-probe", action="store_true", help="Skip HTTP HTML probing")
    parser.add_argument("--skip-semantic", action="store_true", help="Skip ONNX semantic zero-shot scoring")
    parser.add_argument("--skip-rdap", action="store_true", help="Skip RDAP domain age queries")
    parser.add_argument("--concurrency", type=int, default=80)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.stage in ("crux", "all"):
        stage_crux(args.project_id, args.work_dir, args.rank_max)

    if args.stage in ("build", "all"):
        stage_build(
            work_dir=args.work_dir,
            out_dir=args.out_dir,
            allowlist_path=args.allowlist,
            cache_path=args.cache_file,
            rdap_cache_path=args.rdap_cache,
            version=args.version,
            skip_probe=args.skip_probe,
            skip_semantic=args.skip_semantic,
            skip_rdap=args.skip_rdap,
            probe_concurrency=args.concurrency,
        )


if __name__ == "__main__":
    main()
