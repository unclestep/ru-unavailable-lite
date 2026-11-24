"""Offline unit tests for the rustrict pipeline."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import tempfile
import unittest
from pathlib import Path

from rustrict.classifier import DomainClassifier, classify_probed_domains, match_junk_html
from rustrict.crux import CruxSnapshot, filter_by_crux, load_crux_snapshot, save_crux_snapshot
from rustrict.dns_geo import CountryDB, is_ru_hosted
from rustrict.export import (
    build_geoip_lists,
    build_split_rules,
    decode_geoip,
    decode_geosite,
    encode_geoip,
    encode_geosite,
    write_artifacts,
)
from rustrict.filters import (
    apply_deterministic_filters,
    build_stem_counts,
    domain_stem,
    is_mirror_cluster,
    match_junk_domain_rule,
)
from rustrict.normalize import (
    DomainMatcher,
    PSL,
    dedup_by_coverage,
    filter_subdomain_farms,
    normalize_domain,
)
from rustrict.probe import (
    PageProbe,
    classify_probe_outcome,
    is_public_ip,
    load_probe_cache,
    parse_html_metadata,
    save_probe_cache,
)
from rustrict.sources import subtract_cidrs

SAMPLE_PSL = """
// Sample PSL
com
org
net
ru
top
biz
art
co.uk
*.ck
!www.ck
"""


class TestNormalizeAndCrux(unittest.TestCase):
    def setUp(self) -> None:
        self.psl = PSL(SAMPLE_PSL)

    def test_psl_wildcards_and_exceptions(self) -> None:
        self.assertEqual(self.psl.registrable("sub.example.co.uk"), "example.co.uk")
        self.assertEqual(self.psl.registrable("a.b.ck"), "a.b.ck")
        self.assertEqual(self.psl.registrable("www.ck"), "www.ck")

    def test_normalize_domain(self) -> None:
        self.assertEqual(normalize_domain("https://www.Example.COM:443/path?a=1"), "example.com")
        self.assertEqual(normalize_domain("*.sub.test.org."), "sub.test.org")
        self.assertEqual(normalize_domain("пример.рф"), "xn--e1afmkfd.xn--p1ai")
        self.assertIsNone(normalize_domain("1.2.3.4"))
        self.assertIsNone(normalize_domain("1.2.3.4.in-addr.arpa"))
        self.assertIsNone(normalize_domain("ec2-1-2-3-4.compute.amazonaws.com"))

    def test_dedup_by_coverage(self) -> None:
        got = dedup_by_coverage(["a.b.example.com", "example.com", "sub.example.com", "other.org"])
        self.assertEqual(got, ["example.com", "other.org"])

    def test_filter_subdomain_farms_and_protection(self) -> None:
        domains = [f"s{i}.orgs.biz" for i in range(8)] + ["good.orgs.biz", "clean.com", "api.clean.com"]
        protected = DomainMatcher(["good.orgs.biz"])
        kept, farms = filter_subdomain_farms(domains, self.psl, max_subdomains=5, protected=protected)
        self.assertEqual(farms, {"orgs.biz": 9})
        self.assertEqual(sorted(kept), ["api.clean.com", "clean.com", "good.orgs.biz"])

    def test_crux_subdomain_cannot_ride_parent_rank(self) -> None:
        snap = CruxSnapshot(
            yyyymm=202610,
            host_ranks={"orgs.biz": 5000, "music.youtube.com": 1000},
            reg_ranks={"orgs.biz": 5000, "youtube.com": 1000},
        )
        got = filter_by_crux(
            ["youtube.com", "music.youtube.com", "medspravki.orgs.biz", "unknown.com"],
            snap,
            self.psl,
        )
        self.assertEqual(got, ["youtube.com", "music.youtube.com"])

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "crux.json.gz"
            save_crux_snapshot(p, snap)
            loaded = load_crux_snapshot(p)
            self.assertEqual(loaded, snap)


class TestDnsGeo(unittest.TestCase):
    def test_country_db_v4_and_v6(self) -> None:
        v4_csv = "5.255.255.0,5.255.255.255,RU\n1.1.1.0,1.1.1.255,US\n"
        v6_csv = "2a02:6b8::,2a02:6b8:ffff:ffff:ffff:ffff:ffff:ffff,RU\n"
        db = CountryDB(v4_csv, v6_csv)

        self.assertEqual(db.lookup("5.255.255.70"), "RU")
        self.assertEqual(db.lookup("1.1.1.1"), "US")
        self.assertEqual(db.lookup("2a02:6b8::2:242"), "RU")
        self.assertTrue(is_ru_hosted(("5.255.255.70", "2a02:6b8::2:242"), db))
        self.assertFalse(is_ru_hosted(("5.255.255.70", "1.1.1.1"), db))
        self.assertFalse(is_ru_hosted((), db))


class TestFilters(unittest.TestCase):
    def setUp(self) -> None:
        self.psl = PSL(SAMPLE_PSL)

    def test_mirror_cluster_and_allowlist(self) -> None:
        raw = [f"admiralx-{c}{c}{c}.top" for c in "abcdefghijkl"] + [
            "rutracker.org",
            "flibusta.is",
            "meduza.io",
            "onlinepornhub.net",
            "lordfilm6.art",
            "1go.casino",
        ]
        stems = build_stem_counts(raw, self.psl)
        self.assertEqual(domain_stem("admiralx-abc"), "admiralx")
        self.assertTrue(is_mirror_cluster("admiralx-aaa.top", self.psl, stems, min_cluster=10))
        self.assertEqual(match_junk_domain_rule("1go.casino", self.psl), "rule:gambling_tld")
        self.assertEqual(match_junk_domain_rule("lordfilm6.art", self.psl), "rule:piracy_kino")

        kept, dropped = apply_deterministic_filters(
            raw,
            self.psl,
            feed_matcher=DomainMatcher(["badfeed.com"]),
            allow_matcher=DomainMatcher(["rutracker.org", "flibusta.is"]),
            stem_counts=stems,
        )
        self.assertEqual(sorted(kept), ["flibusta.is", "meduza.io", "rutracker.org"])
        self.assertIn("rule:gambling", dropped)
        self.assertIn("rule:porn", dropped)
        self.assertIn("rule:piracy_kino", dropped)


class TestProbeAndSSRF(unittest.TestCase):
    def test_is_public_ip_blocks_private_and_metadata(self) -> None:
        for blocked in (
            "127.0.0.1",
            "169.254.169.254",
            "10.0.0.1",
            "172.16.0.5",
            "192.168.1.1",
            "100.64.0.1",
            "0.0.0.0",
            "::1",
            "fe80::1",
            "fc00::1",
            "::ffff:127.0.0.1",
            "::ffff:169.254.169.254",
        ):
            self.assertFalse(is_public_ip(blocked), f"must reject {blocked}")

        for allowed in ("1.1.1.1", "8.8.8.8", "93.184.216.34", "2606:4700:4700::1111"):
            self.assertTrue(is_public_ip(allowed), f"must allow {allowed}")

    def test_html_parser_and_outcomes(self) -> None:
        html = b"""
        <html><head>
          <title>  Test News  </title>
          <meta name="description" content="Independent journalism">
          <meta name="rating" content="RTA-5042-1996-1400-1577-RTA">
          <script>var bad = "casino slots";</script>
        </head><body><h1>Main Headline</h1><p>Article body text.</p></body></html>
        """
        meta = parse_html_metadata(html)
        self.assertEqual(meta["title"], "Test News")
        self.assertEqual(meta["meta_desc"], "Independent journalism")
        self.assertIn("RTA-5042", meta["meta_rating"])
        self.assertNotIn("casino", meta["snippet"])

        self.assertEqual(classify_probe_outcome(403, {"title": "", "meta_desc": "", "h1": "", "snippet": ""}), "challenge")
        self.assertEqual(
            classify_probe_outcome(200, {"title": "Just a moment...", "meta_desc": "", "h1": "", "snippet": ""}),
            "challenge",
        )
        self.assertEqual(
            classify_probe_outcome(200, {"title": "This domain is for sale", "meta_desc": "", "h1": "", "snippet": ""}),
            "parked",
        )
        self.assertEqual(classify_probe_outcome(404, {"title": "Not Found", "meta_desc": "", "h1": "", "snippet": ""}), "dead")

    def test_probe_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cache.json.gz"
            rec = PageProbe(domain="example.com", status="ok", http_code=200, title="Hi", checked_at=123456)
            save_probe_cache(p, {"example.com": rec})
            loaded = load_probe_cache(p)
            self.assertEqual(loaded["example.com"], rec)


class TestClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.psl = PSL(SAMPLE_PSL)

    def test_match_junk_html_rules(self) -> None:
        feeds = DomainMatcher(["evil-mirror.top"])

        # News mentioning "казино" once must NOT be dropped.
        news = PageProbe(
            domain="news.org",
            status="ok",
            title="Власти обсуждают налоги на онлайн-казино",
            snippet="Экономический обзор недели.",
        )
        self.assertIsNone(match_junk_html(news, self.psl, feeds))

        # Real gambling page with >= 2 signals IS dropped.
        casino = PageProbe(
            domain="lucky-spin.net",
            status="ok",
            title="Онлайн-казино — игровые автоматы и фриспины",
            meta_desc="Получи бонус за регистрацию и крути слоты онлайн",
        )
        self.assertEqual(match_junk_html(casino, self.psl, feeds), "html:gambling")

        # Redirect cloaking to a feed domain IS dropped.
        cloaked = PageProbe(domain="clean-entry.com", status="ok", final_host="evil-mirror.top", title="Welcome")
        self.assertEqual(match_junk_html(cloaked, self.psl, feeds), "html:redirect_feed")

        # Adult RTA meta tag IS dropped.
        adult = PageProbe(domain="tubesite.net", status="ok", meta_rating="RTA-5042-1996-1400-1577-RTA")
        self.assertEqual(match_junk_html(adult, self.psl, feeds), "html:meta_adult")

    def test_domain_and_probed_classification_end_to_end(self) -> None:
        junk_train = [f"vulkancasino{i}.top" for i in range(60)] + [f"free-porn-vid{i}.biz" for i in range(60)]
        legit_train = [f"techjournal{i}.org" for i in range(60)] + [f"bankportal{i}.com" for i in range(60)]

        stems = build_stem_counts(junk_train, self.psl)
        clf = DomainClassifier(self.psl, stems)
        auc = clf.fit(junk_train, legit_train)
        self.assertGreater(auc, 0.95)

        candidates = [
            "rutracker.org",
            "blocked-news.org",
            "parked-site.com",
            "dead-site.com",
            "hidden-casino.net",
            "vulkancasino999.top",
        ]
        probes = {
            "rutracker.org": PageProbe(domain="rutracker.org", status="challenge", http_code=403),
            "blocked-news.org": PageProbe(domain="blocked-news.org", status="challenge", http_code=403),
            "parked-site.com": PageProbe(domain="parked-site.com", status="parked", http_code=200),
            "dead-site.com": PageProbe(domain="dead-site.com", status="dead", http_code=404),
            "hidden-casino.net": PageProbe(
                domain="hidden-casino.net",
                status="ok",
                http_code=200,
                title="Онлайн-казино — игровые автоматы",
                meta_desc="Бонус за регистрацию и фриспины каждый день",
            ),
            "vulkancasino999.top": PageProbe(domain="vulkancasino999.top", status="challenge", http_code=403),
        }

        kept, dropped = classify_probed_domains(
            candidates,
            probes=probes,
            domain_clf=clf,
            psl=self.psl,
            feed_matcher=DomainMatcher([]),
            allow_matcher=DomainMatcher(["rutracker.org"]),
        )
        self.assertEqual(sorted(kept), ["blocked-news.org", "rutracker.org"])
        self.assertIn("hidden-casino.net", dropped.get("html:gambling", []))
        self.assertIn("parked-site.com", dropped.get("html:parked", []))
        self.assertIn("dead-site.com", dropped.get("probe:dead", []))
        self.assertIn("vulkancasino999.top", dropped.get("ml:domain_fallback", []))


class TestExportAndCidrs(unittest.TestCase):
    def test_subtract_cidrs_exact(self) -> None:
        base = [ipaddress.IPv4Network("10.0.0.0/22")]
        cut = [ipaddress.IPv4Network("10.0.1.0/24")]
        got = [str(n) for n in subtract_cidrs(base, cut)]
        self.assertEqual(got, ["10.0.0.0/24", "10.0.2.0/23"])

    def test_export_round_trip_and_split_rules(self) -> None:
        psl = PSL(SAMPLE_PSL)
        foreign = {
            "PRIVATE": [(ipaddress.IPv4Address("10.0.0.0").packed, 8)],
            "TELEGRAM": [
                (ipaddress.IPv4Address("91.108.8.0").packed, 22),
                (ipaddress.IPv4Address("91.108.12.0").packed, 22),
            ],
            "NETFLIX": [(ipaddress.IPv4Address("23.246.0.0").packed, 18)],
            "FACEBOOK": [(ipaddress.IPv4Address("31.13.24.0").packed, 21)],
            "TWITTER": [(ipaddress.IPv4Address("104.244.42.0").packed, 24)],
            "CLOUDFLARE": [(ipaddress.IPv4Address("104.16.0.0").packed, 13)],
            "GOOGLE": [(ipaddress.IPv4Address("34.0.0.0").packed, 9)],  # GCP noise — must be ignored
        }
        google_first_party = [(ipaddress.IPv4Address("142.250.0.0").packed, 15)]

        geoip_lists = build_geoip_lists(
            foreign, google_first_party, ["cloudflare.com", "youtube.com"], psl
        )
        self.assertNotIn("CLOUDFLARE", geoip_lists)
        self.assertEqual(geoip_lists["GOOGLE"], google_first_party)
        # Adjacent Telegram /22 ranges must collapse into /21.
        self.assertEqual(geoip_lists["TELEGRAM"], [(ipaddress.IPv4Address("91.108.8.0").packed, 21)])

        rules = build_split_rules("202611240900", ["meduza.io", "sub.example.com"], ["example.com"], geoip_lists)
        self.assertEqual(rules["domains"], ["domain:example.com", "domain:meduza.io"])
        self.assertNotIn("10.0.0.0/8", rules["ips"])  # PRIVATE excluded from proxy split IPs
        self.assertIn("142.250.0.0/15", rules["ips"])
        self.assertIn("91.108.8.0/21", rules["ips"])

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            write_artifacts(out, "202611240900", ["meduza.io"], ["example.com"], geoip_lists, {"ok": True})

            self.assertEqual(
                decode_geosite((out / "geosite.dat").read_bytes()),
                {"RU-BLOCKED": ["meduza.io"], "RU-FORBIDDEN": ["example.com"]},
            )
            self.assertEqual(decode_geoip((out / "geoip.dat").read_bytes()), geoip_lists)

            parsed_split = json.loads((out / "split-rules.json").read_text(encoding="utf-8"))
            self.assertEqual(parsed_split["version"], "202611240900")

            for line in (out / "checksums.sha256sum").read_text(encoding="utf-8").splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
