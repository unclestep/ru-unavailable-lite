"""Offline unit tests for the rustrict pipeline."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rustrict.classifier import (
    DomainClassifier,
    html_keyword_hits,
    match_hard_html_junk,
)
from rustrict.crux import (
    CruxSnapshot,
    filter_by_crux,
    load_crux_snapshot,
    save_crux_snapshot,
)
from rustrict.dns_geo import CountryDB, DnsResult, is_ru_hosted
from rustrict.export import (
    build_geoip_lists,
    build_split_rules,
    decode_geoip,
    decode_geosite,
    write_artifacts,
)
from rustrict.filters import (
    TaintIndex,
    apply_deterministic_filters,
    build_stem_counts,
    dga_signals,
    domain_stem,
    is_mirror_cluster,
    match_junk_domain_rule,
)
from rustrict.normalize import (
    PSL,
    DomainMatcher,
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
from rustrict.reputation import (
    AsnDB,
    IpBlocklist,
    IpReputation,
    parse_netset,
    parse_rdap_registration_ts,
    parse_spamhaus_asndrop_ndjson,
    parse_spamhaus_drop_ndjson,
)
from rustrict.scoring import score_and_filter_domains
from rustrict.semantic import JUNK_PROTOTYPES, LEGIT_PROTOTYPES, SemanticClassifier
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
cfd
click
co.uk
github.io
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
        self.assertEqual(self.psl.registrable("user.github.io"), "user.github.io")

    def test_normalize_domain(self) -> None:
        self.assertEqual(
            normalize_domain("https://www.Example.COM:443/path?a=1"), "example.com"
        )
        self.assertEqual(normalize_domain("*.sub.test.org."), "sub.test.org")
        self.assertEqual(normalize_domain("пример.рф"), "xn--e1afmkfd.xn--p1ai")
        self.assertIsNone(normalize_domain("1.2.3.4"))
        self.assertIsNone(normalize_domain("1.2.3.4.in-addr.arpa"))
        self.assertIsNone(normalize_domain("ec2-1-2-3-4.compute.amazonaws.com"))

    def test_dedup_by_coverage(self) -> None:
        got = dedup_by_coverage(
            ["a.b.example.com", "example.com", "sub.example.com", "other.org"]
        )
        self.assertEqual(got, ["example.com", "other.org"])

    def test_filter_subdomain_farms_and_protection(self) -> None:
        domains = [f"s{i}.orgs.biz" for i in range(8)] + [
            "good.orgs.biz",
            "clean.com",
            "api.clean.com",
        ]
        protected = DomainMatcher(["good.orgs.biz"])
        kept, farms = filter_subdomain_farms(
            domains, self.psl, max_subdomains=5, protected=protected
        )
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


class TestFiltersAndTaint(unittest.TestCase):
    def setUp(self) -> None:
        self.psl = PSL(SAMPLE_PSL)

    def test_dga_and_leetspeak_signals(self) -> None:
        self.assertTrue(dga_signals("m4rsh-ev4-41.com", self.psl)["leetspeak"])
        self.assertTrue(dga_signals("w3lora17-4udacious.com", self.psl)["leetspeak"])
        self.assertTrue(dga_signals("kzvbjbtgsd.click", self.psl)["consonant_cluster"])
        self.assertFalse(dga_signals("web3.org", self.psl)["leetspeak"])
        self.assertFalse(dga_signals("meduza.io", self.psl)["leetspeak"])

    def test_taint_index_handles_3char_stems_and_private_psl(self) -> None:
        dropped = [
            "cat-ilz24.cfd",
            "cat-woz23.cfd",
            "sub1.badroot.com",
            "evil.github.io",
        ]
        taint = TaintIndex.build(
            dropped, self.psl, allow_matcher=DomainMatcher(["good.com"])
        )

        self.assertTrue(taint.root_is_tainted("sub2.badroot.com", self.psl))
        self.assertEqual(taint.stem_drop_count("cat-cjy43.top", self.psl), 2)
        # Private PSL entries (github.io) isolate tenants: other.github.io must NOT be tainted.
        self.assertFalse(taint.root_is_tainted("other.github.io", self.psl))

    def test_mirror_cluster_and_strict_rules(self) -> None:
        raw = [f"admiralx-{c}{c}{c}.top" for c in "abcdefghijkl"] + [
            "rutracker.org",
            "flibusta.is",
            "meduza.io",
            "onlinepornhub.net",
            "lord-film29gw.top",
            "probivon.top",
            "vapeskidon.site",
            "1go.casino",
        ]
        stems = build_stem_counts(raw, self.psl)
        self.assertEqual(domain_stem("admiralx-abc"), "admiralx")
        self.assertTrue(
            is_mirror_cluster("admiralx-aaa.top", self.psl, stems, min_cluster=10)
        )
        self.assertEqual(
            match_junk_domain_rule("1go.casino", self.psl), "rule:gambling_tld"
        )
        self.assertEqual(
            match_junk_domain_rule("lord-film29gw.top", self.psl), "rule:piracy_kino"
        )
        self.assertEqual(
            match_junk_domain_rule("probivon.top", self.psl), "rule:probiv_leaks"
        )
        self.assertEqual(
            match_junk_domain_rule("vapeskidon.site", self.psl), "rule:illicit_ru"
        )

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
        self.assertIn("rule:probiv_leaks", dropped)
        self.assertIn("rule:illicit_ru", dropped)


class TestReputationAndRDAP(unittest.TestCase):
    def test_ip_asn_blocklists_and_cohosting(self) -> None:
        drop_ndjson = (
            '{"cidr":"198.51.100.0/24","sblid":"SBL1"}\n{"type":"metadata","count":1}\n'
        )
        asndrop_ndjson = (
            '{"asn":64512,"asname":"BULLETPROOF-AS"}\n{"type":"metadata"}\n'
        )
        firehol = "# comment\n203.0.113.0/24\n"
        asn_csv = (
            '1.0.0.0,1.0.0.255,13335,"CLOUDFLARENET - Cloudflare, Inc."\n'
            "45.10.10.0,45.10.10.255,64512,BULLETPROOF-AS\n"
            "91.200.1.0,91.200.1.255,55555,DEDICATED-HOST\n"
        )

        blocklist = IpBlocklist(
            parse_spamhaus_drop_ndjson(drop_ndjson) + parse_netset(firehol)
        )
        bad_asns = parse_spamhaus_asndrop_ndjson(asndrop_ndjson)
        asn_db = AsnDB(asn_csv)
        rep = IpReputation(asn_db, blocklist, bad_asns)

        # Build co-hosting taint: dedicated IP 91.200.1.5 has 3 dropped sites, CDN 1.0.0.1 has 5.
        rep.build_cohosting_taint(
            [("91.200.1.5",)] * 3 + [("1.0.0.1",)] * 5,
            min_junk_per_ip=3,
        )

        self.assertTrue(rep.evaluate(("198.51.100.7",))["bad_ip"])
        self.assertTrue(rep.evaluate(("203.0.113.9",))["bad_ip"])
        self.assertTrue(rep.evaluate(("45.10.10.10",))["bad_asn"])
        self.assertTrue(rep.evaluate(("91.200.1.5",))["cohosted_junk_ip"])
        # Shared CDN ASN (13335) must NEVER be tainted by co-hosting.
        self.assertFalse(rep.evaluate(("1.0.0.1",))["cohosted_junk_ip"])

    def test_parse_rdap_registration_ts(self) -> None:
        payload = {
            "events": [
                {"eventAction": "last changed", "eventDate": "2026-01-01T00:00:00Z"},
                {"eventAction": "registration", "eventDate": "2020-05-15T12:00:00Z"},
            ]
        }
        ts = parse_rdap_registration_ts(payload)
        self.assertIsNotNone(ts)
        self.assertEqual(ts, 1589544000)


class TestProbeAndSSRF(unittest.TestCase):
    def test_is_public_ip_blocks_private_and_metadata(self) -> None:
        for blocked in (
            "127.0.0.1",
            "169.254.169.254",
            "10.0.0.1",
            "172.16.0.5",
            "192.168.1.1",
            "100.64.0.1",
            "0.0.0.0",  # nosec B104
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

        self.assertEqual(
            classify_probe_outcome(
                403, {"title": "", "meta_desc": "", "h1": "", "snippet": ""}
            ),
            "challenge",
        )
        self.assertEqual(
            classify_probe_outcome(
                200,
                {"title": "Just a moment...", "meta_desc": "", "h1": "", "snippet": ""},
            ),
            "challenge",
        )
        self.assertEqual(
            classify_probe_outcome(
                200,
                {
                    "title": "This domain is for sale",
                    "meta_desc": "",
                    "h1": "",
                    "snippet": "",
                },
            ),
            "parked",
        )
        self.assertEqual(
            classify_probe_outcome(
                404, {"title": "Not Found", "meta_desc": "", "h1": "", "snippet": ""}
            ),
            "dead",
        )

    def test_probe_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cache.json.gz"
            rec = PageProbe(
                domain="example.com",
                status="ok",
                http_code=200,
                title="Hi",
                checked_at=123456,
            )
            save_probe_cache(p, {"example.com": rec})
            loaded = load_probe_cache(p)
            self.assertEqual(loaded["example.com"], rec)


class TestSemanticAndScoringEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.psl = PSL(SAMPLE_PSL)

    def test_html_keyword_hits_and_hard_rules(self) -> None:
        feeds = DomainMatcher(["evil-mirror.top"])
        news = PageProbe(
            domain="news.org",
            status="ok",
            title="Власти обсуждают налоги на онлайн-казино",
            snippet="Экономический обзор недели.",
        )
        self.assertEqual(html_keyword_hits(news), ("gambling", 1))
        self.assertIsNone(match_hard_html_junk(news, self.psl, feeds))

        cloaked = PageProbe(
            domain="clean.com", status="ok", final_host="evil-mirror.top", title="Hi"
        )
        self.assertEqual(
            match_hard_html_junk(cloaked, self.psl, feeds), "html:redirect_feed"
        )

        adult = PageProbe(
            domain="tube.net", status="ok", meta_rating="RTA-5042-1996-1400-1577-RTA"
        )
        self.assertEqual(
            match_hard_html_junk(adult, self.psl, feeds), "html:meta_adult"
        )

    def test_unified_scoring_combines_semantic_rdap_mx_taint_and_ip(self) -> None:
        junk_train = [f"vulkancasino{i}.top" for i in range(60)] + [
            f"free-porn-vid{i}.biz" for i in range(60)
        ]
        legit_train = [f"techjournal{i}.org" for i in range(60)] + [
            f"bankportal{i}.com" for i in range(60)
        ]
        stems = build_stem_counts(junk_train, self.psl)
        clf = DomainClassifier(self.psl, stems)
        clf.fit(junk_train, legit_train)

        # Deterministic fake embeddings: [junk_dim, legit_dim]
        def fake_embed(texts: list[str]) -> np.ndarray:
            vecs = []
            for t in texts:
                tl = t.lower()
                if any(
                    w in tl
                    for w in (
                        "казино",
                        "слоты",
                        "порно",
                        "пробив",
                        "сигарет",
                        "вебкам",
                        "онлайн бесплатно",
                    )
                ):
                    vecs.append([0.95, 0.05])
                elif any(
                    w in tl
                    for w in (
                        "новости",
                        "журналист",
                        "software",
                        "форум",
                        "библиотека",
                        "vpn",
                        "обзор",
                    )
                ):
                    vecs.append([0.05, 0.95])
                else:
                    vecs.append([0.5, 0.5])
            return np.asarray(vecs, dtype=np.float32)

        sem_clf = SemanticClassifier(embed_fn=fake_embed)
        self.assertEqual(len(JUNK_PROTOTYPES), 5)
        self.assertEqual(len(LEGIT_PROTOTYPES), 5)

        taint = TaintIndex.build(
            ["cat-a1.cfd", "cat-b2.cfd"], self.psl, DomainMatcher([])
        )
        ip_rep = IpReputation(
            asn_db=AsnDB("91.200.1.0,91.200.1.255,55555,HOST"),
            blocklist=IpBlocklist(["198.51.100.0/24"]),
            bad_asns=set(),
            cohosted_junk_ips={"91.200.1.5": 4},
        )

        candidates = [
            "rutracker.org",  # allowlist -> KEEP
            "news-portal.org",  # 1 casino keyword but legit semantic + top-50k -> KEEP
            "cat-c3.top",  # tainted 3-char stem + sketchy TLD + tail CrUX -> DROP
            "m4rsh-ev4-41.com",  # leetspeak + 403 grey-zone + new RDAP -> DROP
            "mailspam.org",  # web dead + MX record -> DROP (mail_only_spam)
            "cohosted-spam.top",  # dedicated junk IP + sketchy TLD + tail CrUX -> DROP
            "old-borderline.top",  # sketchy TLD + tail CrUX (+50) but >5y RDAP (-40) -> KEEP
            "new-borderline.top",  # sketchy TLD + tail CrUX + weak semantic + <90d RDAP -> DROP
        ]
        probes = {
            "rutracker.org": PageProbe(
                domain="rutracker.org", status="challenge", http_code=403
            ),
            "news-portal.org": PageProbe(
                domain="news-portal.org",
                status="ok",
                http_code=200,
                title="Независимые новости: обзор налогов на онлайн-казино",
            ),
            "cat-c3.top": PageProbe(
                domain="cat-c3.top", status="challenge", http_code=403
            ),
            "m4rsh-ev4-41.com": PageProbe(
                domain="m4rsh-ev4-41.com", status="challenge", http_code=403
            ),
            "mailspam.org": PageProbe(
                domain="mailspam.org", status="dead", http_code=0
            ),
            "cohosted-spam.top": PageProbe(
                domain="cohosted-spam.top", status="challenge", http_code=403
            ),
            "old-borderline.top": PageProbe(
                domain="old-borderline.top",
                status="ok",
                http_code=200,
                title="Old personal project",
            ),
            "new-borderline.top": PageProbe(
                domain="new-borderline.top",
                status="ok",
                http_code=200,
                title="Лучшие слоты и казино бонусы",
            ),
        }
        dns_res = {d: DnsResult("alive", ("93.184.216.34",)) for d in candidates}
        dns_res["cohosted-spam.top"] = DnsResult("alive", ("91.200.1.5",))

        crux_ranks = {
            "rutracker.org": 5_000,
            "news-portal.org": 50_000,
            "cat-c3.top": 500_000,
            "m4rsh-ev4-41.com": 500_000,
            "mailspam.org": 500_000,
            "cohosted-spam.top": 500_000,
            "old-borderline.top": 500_000,
            "new-borderline.top": 500_000,
        }

        def fake_rdap(roots: list[str]) -> dict[str, int]:
            return {
                "m4rsh-ev4-41.com": 20,
                "old-borderline.top": 2200,
                "new-borderline.top": 15,
            }

        kept, results = score_and_filter_domains(
            candidates,
            probes=probes,
            dns_results=dns_res,
            mx_domains={"mailspam.org"},
            crux_ranks=crux_ranks,
            domain_clf=clf,
            semantic_clf=sem_clf,
            taint=taint,
            ip_rep=ip_rep,
            psl=self.psl,
            feed_matcher=DomainMatcher([]),
            allow_matcher=DomainMatcher(["rutracker.org"]),
            rdap_age_fn=fake_rdap,
        )

        by_dom = {r.domain: r for r in results}
        self.assertEqual(
            sorted(kept), ["news-portal.org", "old-borderline.top", "rutracker.org"]
        )
        self.assertEqual(by_dom["mailspam.org"].primary_reason, "mail_only_spam")
        self.assertTrue(by_dom["cat-c3.top"].dropped)
        self.assertTrue(by_dom["m4rsh-ev4-41.com"].dropped)
        self.assertTrue(by_dom["cohosted-spam.top"].dropped)
        self.assertTrue(by_dom["new-borderline.top"].dropped)


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
            "GOOGLE": [(ipaddress.IPv4Address("34.0.0.0").packed, 9)],
        }
        google_first_party = [(ipaddress.IPv4Address("142.250.0.0").packed, 15)]

        geoip_lists = build_geoip_lists(
            foreign, google_first_party, ["cloudflare.com", "youtube.com"], psl
        )
        self.assertNotIn("CLOUDFLARE", geoip_lists)
        self.assertEqual(geoip_lists["GOOGLE"], google_first_party)
        self.assertEqual(
            geoip_lists["TELEGRAM"], [(ipaddress.IPv4Address("91.108.8.0").packed, 21)]
        )

        rules = build_split_rules(
            "202611240900",
            ["meduza.io", "sub.example.com"],
            ["example.com"],
            geoip_lists,
        )
        self.assertEqual(rules["domains"], ["domain:example.com", "domain:meduza.io"])
        self.assertNotIn("10.0.0.0/8", rules["ips"])
        self.assertIn("142.250.0.0/15", rules["ips"])
        self.assertIn("91.108.8.0/21", rules["ips"])

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            write_artifacts(
                out,
                "202611240900",
                ["meduza.io"],
                ["example.com"],
                geoip_lists,
                {"ok": True},
            )

            self.assertEqual(
                decode_geosite((out / "geosite.dat").read_bytes()),
                {"RU-BLOCKED": ["meduza.io"], "RU-FORBIDDEN": ["example.com"]},
            )
            self.assertEqual(
                decode_geoip((out / "geoip.dat").read_bytes()), geoip_lists
            )

            parsed_split = json.loads(
                (out / "split-rules.json").read_text(encoding="utf-8")
            )
            self.assertEqual(parsed_split["version"], "202611240900")

            for line in (
                (out / "checksums.sha256sum").read_text(encoding="utf-8").splitlines()
            ):
                digest, name = line.split("  ", 1)
                self.assertEqual(
                    hashlib.sha256((out / name).read_bytes()).hexdigest(), digest
                )


if __name__ == "__main__":
    unittest.main()
