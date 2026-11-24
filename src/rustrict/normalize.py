"""Domain normalization, Public Suffix List parsing, deduplication, and farm removal."""

from __future__ import annotations

import ipaddress
import re
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

_JUNK_HOST_RE = re.compile(
    r"\.in-addr\.arpa$"
    r"|\.ip6\.arpa$"
    r"|\.(?:ip4|ip6)\.static\."
    r"|\.(?:rev|ptr)\."
    r"|\.(?:amazonaws|googleusercontent|cloudapp|compute)\.com$"
    r"|^[a-f0-9]{16,}\."
)

_DOMAIN_RE = re.compile(
    r"^(?=.{4,253}$)"
    r"(?:[a-z0-9_](?:[a-z0-9_\-]{0,61}[a-z0-9_])?\.)+"
    r"[a-z][a-z0-9\-]{1,62}$"
)


class PSL:
    """Public Suffix List matcher supporting wildcards and exceptions."""

    def __init__(self, text: str) -> None:
        self.rules: set[str] = set()
        self.wildcards: set[str] = set()
        self.exceptions: set[str] = set()

        for line in text.splitlines():
            rule = line.strip()
            if not rule or rule.startswith("//"):
                continue
            if rule.startswith("!"):
                self.exceptions.add(rule[1:])
            elif rule.startswith("*."):
                self.wildcards.add(rule[2:])
            else:
                self.rules.add(rule)

    def public_suffix(self, domain: str) -> str:
        labels = domain.split(".")
        for i in range(len(labels)):
            candidate = ".".join(labels[i:])
            if candidate in self.exceptions:
                return ".".join(labels[i + 1 :])
            if candidate in self.rules:
                return candidate
            parent = ".".join(labels[i + 1 :])
            if parent and parent in self.wildcards:
                return candidate
        return labels[-1]

    def registrable(self, domain: str) -> str:
        suffix = self.public_suffix(domain)
        if domain == suffix:
            return domain
        head = domain[: -len(suffix) - 1]
        return head.rsplit(".", 1)[-1] + "." + suffix

    def sld(self, domain: str) -> str:
        """Return the registrable label before the public suffix."""
        return self.registrable(domain).split(".", 1)[0]

    def is_public_suffix(self, domain: str) -> bool:
        return domain in self.rules or domain in self.wildcards


class DomainMatcher:
    """Matches a domain or any of its parent domains against a set."""

    def __init__(self, domains: Iterable[str]) -> None:
        self.domains = set(domains)

    def match(self, domain: str) -> bool:
        if domain in self.domains:
            return True
        labels = domain.split(".")
        return any(".".join(labels[i:]) in self.domains for i in range(1, len(labels)))


def load_domain_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip().lower()
        if clean and not clean.startswith("#"):
            d = normalize_domain(clean)
            if d:
                out.add(d)
    return out


def normalize_domain(entry: str) -> str | None:
    """Convert raw source entry into a lowercase ASCII hostname, or None if invalid."""
    entry = entry.strip().lower().strip(".")
    if not entry:
        return None
    if "//" in entry:
        entry = urlsplit(entry if "://" in entry else "http://" + entry).netloc
    entry = entry.split("/", 1)[0].split("?", 1)[0]
    if "@" in entry:
        entry = entry.rsplit("@", 1)[-1]
    entry = entry.split(":", 1)[0].strip(".")
    if entry.startswith("*."):
        entry = entry[2:]
    if entry.startswith("www."):
        entry = entry[4:]
    if not entry or "." not in entry:
        return None

    if not entry.isascii():
        try:
            entry = entry.encode("idna").decode("ascii")
        except Exception:
            return None

    try:
        ipaddress.ip_address(entry)
        return None
    except ValueError:
        pass

    if not _DOMAIN_RE.match(entry) or _JUNK_HOST_RE.search(entry):
        return None
    return entry


def dedup_by_coverage(domains: Iterable[str]) -> list[str]:
    """Deduplicate domains and drop subdomains already covered by a parent entry."""
    kept: list[str] = []
    kept_set: set[str] = set()

    for domain in sorted(set(domains), key=lambda d: (d.count("."), d)):
        labels = domain.split(".")
        if any(".".join(labels[i:]) in kept_set for i in range(1, len(labels))):
            continue
        kept.append(domain)
        kept_set.add(domain)

    return sorted(kept)


def filter_subdomain_farms(
    domains: list[str],
    psl: PSL,
    max_subdomains: int = 5,
    protected: DomainMatcher | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Drop multi-subdomain spam/doorway farms (e.g. hundreds of *.orgs.biz or *.lordfilm.art).

    When a registrable root is not blocked itself but has > max_subdomains blocked
    subdomains, it is almost always a spam host or piracy doorway farm.
    """
    sub_counts: Counter[str] = Counter()
    for d in domains:
        reg = psl.registrable(d)
        if d != reg:
            sub_counts[reg] += 1

    farms = {reg: cnt for reg, cnt in sub_counts.items() if cnt > max_subdomains}
    if not farms:
        return domains, {}

    kept: list[str] = []
    for d in domains:
        reg = psl.registrable(d)
        if d != reg and reg in farms and not (protected and protected.match(d)):
            continue
        kept.append(d)

    return kept, farms
