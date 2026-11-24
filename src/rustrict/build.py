"""CLI entrypoint for building rustrict release artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import random
from pathlib import Path

from .classifier import DomainClassifier, classify_probed_domains
from .crux import fetch_crux_snapshot, filter_by_crux, load_crux_snapshot, save_crux_snapshot
from .dns_geo import CountryDB, is_ru_hosted, resolve_domains
from .export import build_geoip_lists, decode_geoip, write_artifacts
from .filters import apply_deterministic_filters, build_stem_counts
from .normalize import (
    DomainMatcher,
    PSL,
    dedup_by_coverage,
    filter_subdomain_farms,
    load_domain_file,
    normalize_domain,
)
from .probe import PageProbe, probe_domains
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


def stage_build(
    work_dir: Path,
    out_dir: Path,
    allowlist_path: Path,
    cache_path: Path | None,
    version: str,
    skip_probe: bool,
    probe_concurrency: int,
) -> None:
    stats: dict = {}

    auth_session = make_session(github_token=os.environ.get("GITHUB_TOKEN"))
    psl = _load_psl(work_dir, auth_session)
    crux = load_crux_snapshot(work_dir / "crux.json.gz")
    stats["crux_yyyymm"] = crux.yyyymm

    # 1. Fetch sources, feeds, GeoIP databases, and Google first-party CIDRs.
    raw = fetch_branch_sources(auth_session)
    junk_feeds = fetch_junk_feeds(auth_session)
    legit_ut1 = fetch_legit_feeds(auth_session)

    geo_v4_csv = auth_session.get(GEOIP_IPV4_URL, timeout=300).text
    geo_v6_csv = auth_session.get(GEOIP_IPV6_URL, timeout=300).text
    country_db = CountryDB(geo_v4_csv, geo_v6_csv)

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

    # Keep raw stem counts across the full RKN registry for mirror-cluster detection.
    stem_counts = build_stem_counts(ru_blocked_all, psl)

    # 3. ru-forbidden: untouched community list except dropping RU-hosted stray entries.
    forb_dns = resolve_domains(ru_forbidden)
    forb_ru = [d for d in ru_forbidden if is_ru_hosted(forb_dns[d].ips, country_db)]
    ru_forbidden = [d for d in ru_forbidden if d not in set(forb_ru)]
    stats["ru_forbidden_dropped_ru_host"] = len(forb_ru)
    log.info("ru-forbidden: kept %d, dropped RU-hosted %d", len(ru_forbidden), len(forb_ru))

    allow_matcher = DomainMatcher(allowlist | set(ru_forbidden))

    # 4. ru-blocked: kill subdomain farms, then filter by CrUX RU popularity.
    ru_blocked, farms = filter_subdomain_farms(ru_blocked_all, psl, max_subdomains=5, protected=allow_matcher)
    stats["subdomain_farms_dropped"] = sum(farms.values())
    log.info("Subdomain farms dropped: %d domains across %d roots", sum(farms.values()), len(farms))

    ru_blocked = filter_by_crux(ru_blocked, crux, psl)
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

    # 6. Deterministic filters: feeds + high-precision RU junk rules + mirror clusters.
    all_junk_feed = set().union(*junk_feeds.values()) if junk_feeds else set()
    feed_matcher = DomainMatcher(all_junk_feed)
    ru_blocked, det_dropped = apply_deterministic_filters(
        ru_blocked, psl, feed_matcher, allow_matcher, stem_counts
    )
    stats["deterministic_dropped"] = {k: len(v) for k, v in det_dropped.items()}
    log.info("ru-blocked after deterministic filters: %d (%s)", len(ru_blocked), stats["deterministic_dropped"])

    # 7. Train Domain+Structural classifier (includes porn, gambling, and RU mirror hits).
    rng = random.Random(42)
    train_junk_set: set[str] = set()
    for key in ("ut1/gambling", "ut1/drogue", "ut1/cryptojacking", "ut1/ddos", "ut1/stalkerware"):
        train_junk_set.update(junk_feeds.get(key, set()))
    for key in ("porn", "gambling_extra"):
        items = sorted(junk_feeds.get(key, set()))
        train_junk_set.update(rng.sample(items, min(len(items), 25_000)))
    for key, items in det_dropped.items():
        if key != "feed":
            train_junk_set.update(items)

    # Exclude adult domains in ru-forbidden from legit training seeds.
    porn_matcher = DomainMatcher(junk_feeds.get("porn", set()))
    clean_forbidden = [d for d in ru_forbidden if not porn_matcher.match(d)]
    train_legit_set = (legit_ut1 | allowlist | set(clean_forbidden)) - train_junk_set
    train_junk_set -= train_legit_set

    domain_clf = DomainClassifier(psl, stem_counts)
    stats["domain_ml_auc"] = round(domain_clf.fit(sorted(train_junk_set), sorted(train_legit_set)), 4)

    # 8. SSRF-safe HTML probe + two-stage HTML/Domain classification.
    probes: dict[str, PageProbe] = {}
    legit_seed_probes: list[PageProbe] = []
    if not skip_probe:
        seed_sample = rng.sample(clean_forbidden, min(len(clean_forbidden), 400))
        to_probe = sorted(set(ru_blocked) | set(seed_sample))
        probes = probe_domains(to_probe, cache_path=cache_path, concurrency=probe_concurrency)
        legit_seed_probes = [
            probes[d]
            for d in seed_sample
            if d in probes and probes[d].status == "ok" and probes[d].combined_text()
        ]

    ru_blocked, ml_dropped = classify_probed_domains(
        ru_blocked,
        probes=probes,
        domain_clf=domain_clf,
        psl=psl,
        feed_matcher=feed_matcher,
        allow_matcher=allow_matcher,
        legit_seed_probes=legit_seed_probes,
    )
    stats["ml_probe_dropped"] = {k: len(v) for k, v in ml_dropped.items()}
    log.info("ru-blocked after HTML+ML: %d (%s)", len(ru_blocked), stats["ml_probe_dropped"])

    # 9. Final deduplication and export (ru-forbidden takes priority over ru-blocked).
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
    parser.add_argument("--version", default=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M"))
    parser.add_argument("--skip-probe", action="store_true", help="Skip HTTP HTML probing (offline/fast mode)")
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
            version=args.version,
            skip_probe=args.skip_probe,
            probe_concurrency=args.concurrency,
        )


if __name__ == "__main__":
    main()
