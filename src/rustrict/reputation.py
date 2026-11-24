"""IP/ASN reputation (Spamhaus DROP/ASN-DROP, FireHOL, co-hosting taint) and RDAP domain age."""

from __future__ import annotations

import asyncio
import bisect
import csv
import datetime as dt
import gzip
import io
import ipaddress
import json
import logging
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import aiohttp
import requests

from .sources import (
    FIREHOL_L1_URL,
    FIREHOL_L2_URL,
    IPTOASN_V4_URL,
    IPTOASN_V6_URL,
    SPAMHAUS_ASNDROP_URL,
    SPAMHAUS_DROP_V4_URL,
    SPAMHAUS_DROP_V6_URL,
)

log = logging.getLogger(__name__)

# Shared CDN / cloud / DDoS-protection ASNs excluded from dedicated-IP co-hosting taint.
SHARED_CDN_ASNS = frozenset(
    {
        13335,  # Cloudflare
        54113,  # Fastly
        16509,  # Amazon AWS / CloudFront
        14618,  # Amazon AWS
        15169,  # Google
        396982,  # Google Cloud
        8075,  # Microsoft Azure
        20940,  # Akamai
        16625,  # Akamai
        32934,  # Meta
        19551,  # Imperva / Incapsula
        57724,  # DDoS-Guard
        132892,  # Cloudflare China
    }
)

RDAP_URL = "https://rdap.org/domain/{}"
RDAP_CACHE_TTL_SEC = 90 * 86_400
RDAP_MISS_TTL_SEC = 14 * 86_400


class AsnDB:
    """Binary-search IPv4/IPv6-to-ASN lookup backed by iptoasn CSV files."""

    def __init__(self, ipv4_csv: str = "", ipv6_csv: str = "") -> None:
        self._v4 = (
            self._parse(ipv4_csv, ipaddress.IPv4Address) if ipv4_csv else ([], [], [])
        )
        self._v6 = (
            self._parse(ipv6_csv, ipaddress.IPv6Address) if ipv6_csv else ([], [], [])
        )

    @staticmethod
    def _parse(csv_text: str, addr_cls: type) -> tuple[list[int], list[int], list[int]]:
        rows: list[tuple[int, int, int]] = []
        for parts in csv.reader(io.StringIO(csv_text)):
            if len(parts) < 3:
                continue
            try:
                start = int(addr_cls(parts[0].strip()))
                end = int(addr_cls(parts[1].strip()))
                asn = int(parts[2].strip())
            except (ipaddress.AddressValueError, ValueError):
                continue
            if asn > 0:
                rows.append((start, end, asn))

        rows.sort()
        return (
            [r[0] for r in rows],
            [r[1] for r in rows],
            [r[2] for r in rows],
        )

    def lookup(self, ip: str) -> int | None:
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return None

        starts, ends, asns = self._v4 if addr.version == 4 else self._v6
        if not starts:
            return None

        val = int(addr)
        idx = bisect.bisect_right(starts, val) - 1
        if idx >= 0 and val <= ends[idx]:
            return asns[idx]
        return None


class IpBlocklist:
    """Binary-search IPv4/IPv6 blocklist built from CIDR strings."""

    def __init__(self, cidrs: Iterable[str] = ()) -> None:
        v4_nets: list[ipaddress.IPv4Network] = []
        v6_nets: list[ipaddress.IPv6Network] = []
        for raw in cidrs:
            try:
                net = ipaddress.ip_network(raw.strip(), strict=False)
            except ValueError:
                continue
            if net.version == 4:
                v4_nets.append(net)
            else:
                v6_nets.append(net)

        self._v4 = self._to_intervals(v4_nets)
        self._v6 = self._to_intervals(v6_nets)

    @staticmethod
    def _to_intervals(nets: list) -> tuple[list[int], list[int]]:
        if not nets:
            return [], []
        collapsed = list(ipaddress.collapse_addresses(nets))
        return (
            [int(n.network_address) for n in collapsed],
            [int(n.broadcast_address) for n in collapsed],
        )

    def contains(self, ip: str) -> bool:
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False

        starts, ends = self._v4 if addr.version == 4 else self._v6
        if not starts:
            return False

        val = int(addr)
        idx = bisect.bisect_right(starts, val) - 1
        return idx >= 0 and val <= ends[idx]


class IpReputation:
    """Combines public IP/ASN blocklists with dedicated-IP co-hosting taint."""

    def __init__(
        self,
        asn_db: AsnDB,
        blocklist: IpBlocklist,
        bad_asns: set[int],
        cohosted_junk_ips: dict[str, int] | None = None,
    ) -> None:
        self.asn_db = asn_db
        self.blocklist = blocklist
        self.bad_asns = bad_asns
        self.cohosted_junk_ips = cohosted_junk_ips or {}

    def build_cohosting_taint(
        self,
        dropped_domain_ips: Iterable[tuple[str, ...]],
        min_junk_per_ip: int = 3,
    ) -> dict[str, int]:
        """Flag non-CDN dedicated IPs hosting >= min_junk_per_ip already-dropped domains."""
        counts: Counter[str] = Counter()
        for ips in dropped_domain_ips:
            for ip in set(ips):
                asn = self.asn_db.lookup(ip)
                if asn in SHARED_CDN_ASNS:
                    continue
                counts[ip] += 1

        self.cohosted_junk_ips = {
            ip: cnt for ip, cnt in counts.items() if cnt >= min_junk_per_ip
        }
        return self.cohosted_junk_ips

    def evaluate(self, ips: tuple[str, ...]) -> dict[str, bool]:
        """Check whether resolved IPs hit bad IP ranges, bad ASNs, or junk co-hosting."""
        if not ips:
            return {"bad_ip": False, "bad_asn": False, "cohosted_junk_ip": False}

        bad_ip = any(self.blocklist.contains(ip) for ip in ips)
        bad_asn = False
        cohosted = False
        for ip in ips:
            asn = self.asn_db.lookup(ip)
            if asn is not None and asn in self.bad_asns:
                bad_asn = True
            if ip in self.cohosted_junk_ips:
                cohosted = True

        return {"bad_ip": bad_ip, "bad_asn": bad_asn, "cohosted_junk_ip": cohosted}


def parse_spamhaus_drop_ndjson(text: str) -> list[str]:
    cidrs: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "cidr" in obj:
            cidrs.append(obj["cidr"])
    return cidrs


def parse_spamhaus_asndrop_ndjson(text: str) -> set[int]:
    asns: set[int] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("asn"), int):
            asns.add(obj["asn"])
    return asns


def parse_netset(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def fetch_ip_reputation(session) -> IpReputation:
    """Download Spamhaus DROP/ASN-DROP, FireHOL L1/L2, and iptoasn databases."""
    cidrs: list[str] = []
    bad_asns: set[int] = set()

    for url in (SPAMHAUS_DROP_V4_URL, SPAMHAUS_DROP_V6_URL):
        try:
            cidrs.extend(parse_spamhaus_drop_ndjson(session.get(url, timeout=60).text))
        except (requests.RequestException, ValueError) as exc:
            log.warning("Spamhaus DROP (%s) failed: %s", url, exc)

    try:
        bad_asns = parse_spamhaus_asndrop_ndjson(
            session.get(SPAMHAUS_ASNDROP_URL, timeout=60).text
        )
    except (requests.RequestException, ValueError) as exc:
        log.warning("Spamhaus ASN-DROP failed: %s", exc)

    for url in (FIREHOL_L1_URL, FIREHOL_L2_URL):
        try:
            cidrs.extend(parse_netset(session.get(url, timeout=60).text))
        except (requests.RequestException, ValueError) as exc:
            log.warning("FireHOL (%s) failed: %s", url, exc)

    v4_csv = v6_csv = ""
    try:
        v4_csv = session.get(IPTOASN_V4_URL, timeout=180).text
    except requests.RequestException as exc:
        log.warning("iptoasn IPv4 failed: %s", exc)
    try:
        v6_csv = session.get(IPTOASN_V6_URL, timeout=180).text
    except requests.RequestException as exc:
        log.warning("iptoasn IPv6 failed: %s", exc)

    blocklist = IpBlocklist(cidrs)
    asn_db = AsnDB(v4_csv, v6_csv)
    log.info("IP reputation loaded: %d cidrs, %d bad ASNs", len(cidrs), len(bad_asns))
    return IpReputation(asn_db=asn_db, blocklist=blocklist, bad_asns=bad_asns)


def parse_rdap_registration_ts(payload: dict) -> int | None:
    """Extract Unix timestamp of domain registration event from an RDAP JSON response."""
    for event in payload.get("events", ()):
        if str(event.get("eventAction", "")).lower() == "registration":
            raw_date = event.get("eventDate", "")
            if not raw_date:
                continue
            try:
                parsed = dt.datetime.fromisoformat(raw_date)
                return int(parsed.timestamp())
            except ValueError:
                continue
    return None


def _load_rdap_cache(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _save_rdap_cache(path: Path | None, cache: dict[str, dict]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(cache, f)


async def _fetch_rdap_batch_async(
    roots: list[str], concurrency: int = 12
) -> dict[str, int | None]:
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=6.0, sock_connect=3.0)
    headers = {
        "User-Agent": "rustrict",
        "Accept": "application/rdap+json, application/json",
    }

    async def query_one(
        sess: aiohttp.ClientSession, root: str
    ) -> tuple[str, int | None]:
        async with sem:
            try:
                async with sess.get(
                    RDAP_URL.format(root), allow_redirects=True
                ) as resp:
                    if resp.status != 200:
                        return root, None
                    data = await resp.json(content_type=None)
                    return root, parse_rdap_registration_ts(data)
            except (aiohttp.ClientError, TimeoutError, OSError, ValueError):
                return root, None

    out: dict[str, int | None] = {}
    async with aiohttp.ClientSession(
        timeout=timeout,
        headers=headers,
        cookie_jar=aiohttp.DummyCookieJar(),
    ) as session:
        tasks = [asyncio.create_task(query_one(session, r)) for r in roots]
        for coro in asyncio.as_completed(tasks):
            root, reg_ts = await coro
            out[root] = reg_ts
    return out


def fetch_domain_ages_days(
    registrable_roots: Iterable[str],
    cache_path: Path | None = None,
    concurrency: int = 12,
    max_queries: int = 600,
) -> dict[str, int]:
    """Return domain registration age in days via RDAP (cached, capped for CI speed)."""
    now = int(time.time())
    cache = _load_rdap_cache(cache_path)

    unique_roots = sorted(set(registrable_roots))
    ages: dict[str, int] = {}
    to_query: list[str] = []

    for root in unique_roots:
        rec = cache.get(root)
        if rec:
            ttl = (
                RDAP_CACHE_TTL_SEC
                if rec.get("reg_ts") is not None
                else RDAP_MISS_TTL_SEC
            )
            if (now - int(rec.get("checked_at", 0))) < ttl:
                if rec.get("reg_ts") is not None:
                    ages[root] = max(0, (now - int(rec["reg_ts"])) // 86_400)
                continue
        to_query.append(root)

    if len(to_query) > max_queries:
        log.info("RDAP capped at %d of %d borderline roots", max_queries, len(to_query))
        to_query = to_query[:max_queries]

    log.info("RDAP domain age: %d cached, %d to query", len(ages), len(to_query))
    if to_query:
        fetched = asyncio.run(
            _fetch_rdap_batch_async(to_query, concurrency=concurrency)
        )
        for root, reg_ts in fetched.items():
            cache[root] = {"reg_ts": reg_ts, "checked_at": now}
            if reg_ts is not None:
                ages[root] = max(0, (now - reg_ts) // 86_400)
        _save_rdap_cache(cache_path, cache)

    return ages
