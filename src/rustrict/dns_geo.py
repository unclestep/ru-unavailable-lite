"""Single-pass DNS resolution (A + AAAA) and IP-to-country hosting checks."""

from __future__ import annotations

import bisect
import concurrent.futures as cf
import ipaddress
import logging
from dataclasses import dataclass

import dns.exception
import dns.resolver

log = logging.getLogger(__name__)

NAMESERVERS = ["1.1.1.1", "8.8.8.8", "9.9.9.9"]
FAST_TIMEOUT = 2.0
SLOW_TIMEOUT = 10.0


@dataclass(frozen=True)
class DnsResult:
    status: str  # "alive", "dead", "unknown"
    ips: tuple[str, ...]


class CountryDB:
    """Binary-search IP-to-country lookup supporting both IPv4 and IPv6 ranges."""

    def __init__(self, ipv4_csv: str, ipv6_csv: str = "") -> None:
        self._v4 = self._parse_csv(ipv4_csv, ipaddress.IPv4Address)
        self._v6 = (
            self._parse_csv(ipv6_csv, ipaddress.IPv6Address)
            if ipv6_csv
            else ([], [], [])
        )

    @staticmethod
    def _parse_csv(
        csv_text: str, addr_cls: type
    ) -> tuple[list[int], list[int], list[str]]:
        rows: list[tuple[int, int, str]] = []
        for line in csv_text.splitlines():
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                start = int(addr_cls(parts[0].strip()))
                end = int(addr_cls(parts[1].strip()))
            except ipaddress.AddressValueError:
                continue
            rows.append((start, end, parts[2].strip().upper()))

        rows.sort()
        return (
            [r[0] for r in rows],
            [r[1] for r in rows],
            [r[2] for r in rows],
        )

    def lookup(self, ip: str) -> str | None:
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return None

        starts, ends, codes = self._v4 if addr.version == 4 else self._v6
        if not starts:
            return None

        val = int(addr)
        idx = bisect.bisect_right(starts, val) - 1
        if idx >= 0 and val <= ends[idx]:
            return codes[idx]
        return None


def _new_resolver() -> dns.resolver.Resolver:
    r = dns.resolver.Resolver(configure=False)
    r.nameservers = list(NAMESERVERS)
    return r


def _lookup_one(
    resolver: dns.resolver.Resolver, domain: str, timeout: float
) -> DnsResult:
    ips: list[str] = []
    nxdomain = False
    had_error = False

    for rdtype in ("A", "AAAA"):
        try:
            answers = resolver.resolve(domain, rdtype, lifetime=timeout)
            ips.extend(str(r) for r in answers)
        except dns.resolver.NXDOMAIN:
            nxdomain = True
            break
        except dns.resolver.NoAnswer:
            continue
        except (dns.exception.DNSException, OSError):
            had_error = True

    if ips:
        return DnsResult("alive", tuple(ips[:6]))
    if nxdomain:
        return DnsResult("dead", ())
    if had_error:
        return DnsResult("unknown", ())
    return DnsResult("dead", ())


def resolve_domains(domains: list[str], workers: int = 100) -> dict[str, DnsResult]:
    """Resolve A/AAAA records in a single pass with a slow retry for timeouts."""
    resolver = _new_resolver()
    results: dict[str, DnsResult] = {}

    def run_pass(batch: list[str], timeout: float) -> None:
        if not batch:
            return
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            for d, res in zip(
                batch, pool.map(lambda x: _lookup_one(resolver, x, timeout), batch)
            ):
                results[d] = res

    run_pass(domains, FAST_TIMEOUT)
    unknown = [d for d, r in results.items() if r.status == "unknown"]
    if unknown:
        log.info("DNS retry for %d timed-out domains", len(unknown))
        run_pass(unknown, SLOW_TIMEOUT)

    return results


def check_mx_records(
    domains: list[str], workers: int = 80, timeout: float = FAST_TIMEOUT
) -> set[str]:
    """Return subset of domains that publish MX records (useful to flag mail-only domains)."""
    if not domains:
        return set()

    resolver = _new_resolver()

    def has_mx(domain: str) -> bool:
        try:
            answers = resolver.resolve(domain, "MX", lifetime=timeout)
            return len(answers) > 0
        except (dns.exception.DNSException, OSError):
            return False

    found: set[str] = set()
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for d, ok in zip(domains, pool.map(has_mx, domains)):
            if ok:
                found.add(d)
    return found


def is_ru_hosted(ips: tuple[str, ...], db: CountryDB) -> bool:
    """Return True if all resolved IPs belong to Russia (mixed CDN results stay kept)."""
    if not ips:
        return False
    codes = [db.lookup(ip) for ip in ips]
    return all(cc == "RU" for cc in codes)
