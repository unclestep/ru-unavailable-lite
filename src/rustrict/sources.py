"""HTTP fetchers for blocklists, community lists, threat feeds, PSL, and Google IP ranges."""

from __future__ import annotations

import ipaddress
import logging
import re
import time
from typing import Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

BLOCKED_SOURCES = {
    "antifilter": "https://antifilter.download/list/domains.lst",
}

FORBIDDEN_SOURCES = {
    "antifilter_community": "https://community.antifilter.download/list/domains.lst",
    "re_filter_community": "https://raw.githubusercontent.com/1andrevich/Re-filter-lists/refs/heads/main/community.lst",
    "dartraiden": "https://raw.githubusercontent.com/dartraiden/no-russia-hosts/refs/heads/master/hosts.txt",
    "itdoginfo": "https://raw.githubusercontent.com/itdoginfo/allow-domains/refs/heads/main/Categories/geoblock.lst",
    "internet_helper": "https://raw.githubusercontent.com/Internet-Helper/Unblock-for-Russia/refs/heads/main/geoblock.lst",
}

VERNETTE_API_URL = "https://api.github.com/repos/vernette/rulesets/contents/raw"
VERNETTE_RAW_BASE = "https://raw.githubusercontent.com/vernette/rulesets/master/raw/"

PSL_URL = "https://raw.githubusercontent.com/publicsuffix/list/master/public_suffix_list.dat"

GEOIP_IPV4_URL = "https://github.com/sapics/ip-location-db/releases/download/latest/server-country-ipv4.csv"
GEOIP_IPV6_URL = "https://github.com/sapics/ip-location-db/releases/download/latest/server-country-ipv6.csv"

FOREIGN_GEOIP_URL = "https://github.com/Loyalsoldier/v2ray-rules-dat/releases/latest/download/geoip.dat"
GOOGLE_ALL_IPS_URL = "https://www.gstatic.com/ipranges/goog.json"
GOOGLE_CLOUD_IPS_URL = "https://www.gstatic.com/ipranges/cloud.json"

UT1_BASE = "https://raw.githubusercontent.com/olbat/ut1-blacklists/master/blacklists/{}/domains"
UT1_JUNK_CATS = ("gambling", "drogue", "cryptojacking", "ddos", "stalkerware", "malware")
UT1_LEGIT_CATS = ("shopping", "bank", "press", "jobsearch", "games")

PORN_HOSTS_URLS = (
    "https://raw.githubusercontent.com/StevenBlack/hosts/master/extensions/porn/sinfonietta/hosts",
    "https://raw.githubusercontent.com/StevenBlack/hosts/master/extensions/porn/clefspeare13/hosts",
)
GAMBLING_HOSTS_URLS = (
    "https://raw.githubusercontent.com/StevenBlack/hosts/master/alternates/gambling-only/hosts",
)
HAGEZI_GAMBLING_URL = "https://raw.githubusercontent.com/hagezi/dns-blocklists/main/wildcard/gambling-onlydomains.txt"
ADGUARD_PHISHING_URL = (
    "https://raw.githubusercontent.com/AdguardTeam/HostlistsRegistry/main/"
    "filters/security/filter_30_PhishingURLBlocklist/filter.txt"
)

_HOSTS_RE = re.compile(r"^(?:0\.0\.0\.0|127\.0\.0\.1)\s+([a-z0-9.\-]+)")
_ADBLOCK_RE = re.compile(r"^\|\|([a-z0-9.\-]+)\^")


def make_session(github_token: str | None = None) -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = "rustrict"
    if github_token:
        session.headers["Authorization"] = f"Bearer {github_token}"

    retry = Retry(total=3, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_plain_list(session: requests.Session, url: str, timeout: int = 120) -> list[str]:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return [
        line.strip().lower()
        for line in resp.text.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "!", "//"))
    ]


def fetch_hosts_file(session: requests.Session, url: str, timeout: int = 120) -> list[str]:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    out = []
    for line in resp.text.splitlines():
        m = _HOSTS_RE.match(line.strip().lower())
        if m and m.group(1) not in ("0.0.0.0", "localhost"):
            out.append(m.group(1))
    return out


def fetch_adblock_list(session: requests.Session, url: str, timeout: int = 120) -> set[str]:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    out = set()
    for line in resp.text.splitlines():
        m = _ADBLOCK_RE.match(line.strip().lower())
        if m:
            out.add(m.group(1))
    return out


def fetch_vernette_rulesets(session: requests.Session, timeout: int = 120) -> list[str]:
    resp = session.get(VERNETTE_API_URL, timeout=timeout)
    resp.raise_for_status()
    entries = resp.json()
    if not isinstance(entries, list):
        raise RuntimeError(f"Unexpected GitHub API response: {entries}")

    domains: set[str] = set()
    for entry in entries:
        name = entry.get("name", "")
        if not name.endswith(".txt") or name == "rkn.txt":
            continue
        try:
            domains.update(fetch_plain_list(session, VERNETTE_RAW_BASE + name))
        except Exception as exc:
            log.warning("vernette/%s skipped: %s", name, exc)
        time.sleep(0.15)
    return sorted(domains)


def fetch_branch_sources(session: requests.Session) -> dict[str, list[str]]:
    raw = {"ru-blocked": [], "ru-forbidden": []}

    for name, url in BLOCKED_SOURCES.items():
        items = fetch_plain_list(session, url, timeout=300)
        raw["ru-blocked"].extend(items)
        log.info("source ru-blocked/%s: %d", name, len(items))

    for name, url in FORBIDDEN_SOURCES.items():
        try:
            items = fetch_plain_list(session, url)
            raw["ru-forbidden"].extend(items)
            log.info("source ru-forbidden/%s: %d", name, len(items))
        except Exception as exc:
            log.warning("source ru-forbidden/%s failed: %s", name, exc)

    try:
        items = fetch_vernette_rulesets(session)
        raw["ru-forbidden"].extend(items)
        log.info("source ru-forbidden/vernette: %d", len(items))
    except Exception as exc:
        log.warning("source ru-forbidden/vernette failed: %s", exc)

    return raw


def fetch_junk_feeds(session: requests.Session) -> dict[str, set[str]]:
    """Fetch categorized junk domain feeds (gambling, porn, phishing, malware, etc.)."""
    feeds: dict[str, set[str]] = {}

    for cat in UT1_JUNK_CATS:
        try:
            feeds[f"ut1/{cat}"] = set(fetch_plain_list(session, UT1_BASE.format(cat)))
            log.info("feed ut1/%s: %d", cat, len(feeds[f"ut1/{cat}"]))
        except Exception as exc:
            log.warning("feed ut1/%s failed: %s", cat, exc)

    porn: set[str] = set()
    for url in PORN_HOSTS_URLS:
        try:
            porn.update(fetch_hosts_file(session, url))
        except Exception as exc:
            log.warning("feed porn (%s) failed: %s", url, exc)
    feeds["porn"] = porn
    log.info("feed porn: %d", len(porn))

    gambling_extra: set[str] = set()
    for url in GAMBLING_HOSTS_URLS:
        try:
            gambling_extra.update(fetch_hosts_file(session, url))
        except Exception as exc:
            log.warning("feed gambling (%s) failed: %s", url, exc)
    try:
        gambling_extra.update(fetch_plain_list(session, HAGEZI_GAMBLING_URL))
    except Exception as exc:
        log.warning("feed hagezi/gambling failed: %s", exc)
    feeds["gambling_extra"] = gambling_extra
    log.info("feed gambling_extra: %d", len(gambling_extra))

    try:
        feeds["adguard/phishing"] = fetch_adblock_list(session, ADGUARD_PHISHING_URL)
        log.info("feed adguard/phishing: %d", len(feeds["adguard/phishing"]))
    except Exception as exc:
        log.warning("feed adguard/phishing failed: %s", exc)

    return feeds


def fetch_legit_feeds(session: requests.Session) -> set[str]:
    legit: set[str] = set()
    for cat in UT1_LEGIT_CATS:
        try:
            legit.update(fetch_plain_list(session, UT1_BASE.format(cat)))
        except Exception as exc:
            log.warning("legit feed ut1/%s failed: %s", cat, exc)
    log.info("legit UT1 domains: %d", len(legit))
    return legit


def subtract_cidrs(
    base: Iterable[ipaddress.IPv4Network | ipaddress.IPv6Network],
    cut: Iterable[ipaddress.IPv4Network | ipaddress.IPv6Network],
) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Return collapsed CIDR networks in `base` excluding all addresses in `cut`."""
    res = list(ipaddress.collapse_addresses(base))
    cut_collapsed = list(ipaddress.collapse_addresses(cut))
    for c in cut_collapsed:
        nxt = []
        for b in res:
            if not b.overlaps(c):
                nxt.append(b)
            elif c.subnet_of(b):
                nxt.extend(b.address_exclude(c))
            # If b is a subnet of c, b is completely excluded.
        res = nxt
    return list(ipaddress.collapse_addresses(res))


def fetch_google_first_party_cidrs(session: requests.Session, timeout: int = 60) -> list[tuple[bytes, int]]:
    """Compute Google's own infrastructure CIDRs (goog.json minus GCP customer cloud.json)."""
    goog_prefixes = session.get(GOOGLE_ALL_IPS_URL, timeout=timeout).json()["prefixes"]
    cloud_prefixes = session.get(GOOGLE_CLOUD_IPS_URL, timeout=timeout).json()["prefixes"]

    def split_v4_v6(items: list[dict]) -> tuple[list[ipaddress.IPv4Network], list[ipaddress.IPv6Network]]:
        v4, v6 = [], []
        for p in items:
            if "ipv4Prefix" in p:
                v4.append(ipaddress.IPv4Network(p["ipv4Prefix"]))
            elif "ipv6Prefix" in p:
                v6.append(ipaddress.IPv6Network(p["ipv6Prefix"]))
        return v4, v6

    g4, g6 = split_v4_v6(goog_prefixes)
    c4, c6 = split_v4_v6(cloud_prefixes)

    net4 = subtract_cidrs(g4, c4)
    net6 = subtract_cidrs(g6, c6)
    out = [(n.network_address.packed, n.prefixlen) for n in net4]
    out.extend((n.network_address.packed, n.prefixlen) for n in net6)
    log.info("Google first-party CIDRs (goog - cloud): %d v4 + %d v6 = %d", len(net4), len(net6), len(out))
    return out
