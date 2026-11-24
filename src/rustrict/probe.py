"""SSRF-safe asynchronous HTML metadata crawler with gzipped JSON cache."""

from __future__ import annotations

import asyncio
import gzip
import ipaddress
import json
import logging
import re
import socket
import time
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import aiohttp
from aiohttp.abc import AbstractResolver

log = logging.getLogger(__name__)

MAX_REDIRECTS = 3
MAX_BODY_BYTES = 65_536
CHUNK_BYTES = 8_192
CONNECT_TIMEOUT = 3.0
TOTAL_TIMEOUT = 6.0

CACHE_TTL_OK_SEC = 30 * 86_400
CACHE_TTL_DEAD_SEC = 7 * 86_400

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

_CHALLENGE_RE = re.compile(
    r"(?:just a moment\.\.\.|attention required!|checking your browser|"
    r"ddos-guard|enable javascript and cookies|cf-browser-verification|"
    r"please wait while we verify|access denied|bot protection)",
    re.IGNORECASE,
)

_PARKED_RE = re.compile(
    r"(?:this domain (?:is for sale|has expired|may be for sale)|"
    r"domain (?:is )?(?:for sale|parked)|buy this domain|"
    r"sedoparking|dan\.com|afternic|hugedomains|bodis\.com|"
    r"домен (?:прода[её]тся|зарегистрирован и припаркован|свободен)|"
    r"припаркован в reg\.ru|купить этот домен)",
    re.IGNORECASE,
)

_WS_RE = re.compile(r"\s+")


@dataclass
class PageProbe:
    domain: str
    status: str  # "ok", "challenge", "parked", "dead", "blocked_ssrf", "non_html"
    http_code: int = 0
    final_host: str = ""
    title: str = ""
    meta_desc: str = ""
    meta_keywords: str = ""
    meta_rating: str = ""
    h1: str = ""
    snippet: str = ""
    checked_at: int = 0

    def combined_text(self) -> str:
        parts = [self.title, self.meta_desc, self.meta_keywords, self.h1, self.snippet]
        return " ".join(p for p in parts if p).strip()


def is_public_ip(ip_str: str) -> bool:
    """Return True only for globally routable unicast IPs (blocks loopback/LAN/metadata/CGNAT)."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False

    if addr.version == 6 and addr.ipv4_mapped:
        addr = addr.ipv4_mapped

    return (
        addr.is_global
        and not addr.is_private
        and not addr.is_loopback
        and not addr.is_link_local
        and not addr.is_multicast
        and not addr.is_reserved
        and not addr.is_unspecified
    )


class SSRFBlockedError(Exception):
    """Raised when a hostname resolves to a non-public IP address."""


class _SafeResolver(AbstractResolver):
    """Async DNS resolver that rejects any non-public IP before TCP connect()."""

    def __init__(self) -> None:
        self._inner = aiohttp.DefaultResolver()

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_UNSPEC,
    ) -> list[dict]:
        infos = await self._inner.resolve(host, port, family=family)
        safe = [info for info in infos if is_public_ip(info["host"])]
        if not safe:
            raise SSRFBlockedError(f"{host} resolved only to non-public IPs")
        return safe

    async def close(self) -> None:
        await self._inner.close()


class _MetadataParser(HTMLParser):
    """Pure-stdlib HTML extractor for title, meta tags, h1, and visible text snippet."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.h1_parts: list[str] = []
        self.body_parts: list[str] = []
        self.meta_desc = ""
        self.meta_keywords = ""
        self.meta_rating = ""
        self._in_title = False
        self._in_h1 = False
        self._skip_depth = 0
        self._h1_count = 0
        self._body_len = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in ("script", "style", "noscript", "svg", "template"):
            self._skip_depth += 1
        elif t == "title":
            self._in_title = True
        elif t == "h1" and self._h1_count < 2:
            self._in_h1 = True
        elif t == "meta":
            attr_map = {k.lower(): (v or "") for k, v in attrs}
            name = (
                (attr_map.get("name") or attr_map.get("property") or "").lower().strip()
            )
            content = attr_map.get("content", "").strip()
            if not content:
                return
            if name in ("description", "og:description") and not self.meta_desc:
                self.meta_desc = content[:400]
            elif name == "keywords" and not self.meta_keywords:
                self.meta_keywords = content[:300]
            elif name in ("rating", "rta") and not self.meta_rating:
                self.meta_rating = content[:100]

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if (
            t in ("script", "style", "noscript", "svg", "template")
            and self._skip_depth > 0
        ):
            self._skip_depth -= 1
        elif t == "title":
            self._in_title = False
        elif t == "h1" and self._in_h1:
            self._in_h1 = False
            self._h1_count += 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        elif self._in_h1:
            self.h1_parts.append(text)
        elif self._body_len < 800:
            self.body_parts.append(text)
            self._body_len += len(text) + 1


def parse_html_metadata(
    html_bytes: bytes, charset: str | None = None
) -> dict[str, str]:
    text = html_bytes.decode(charset or "utf-8", errors="replace")
    parser = _MetadataParser()
    try:
        parser.feed(text)
    except (ValueError, AssertionError) as exc:
        log.debug("HTML parser stopped early: %s", exc)

    def clean(parts: list[str], limit: int) -> str:
        return _WS_RE.sub(" ", " ".join(parts)).strip()[:limit]

    return {
        "title": clean(parser.title_parts, 300),
        "meta_desc": _WS_RE.sub(" ", parser.meta_desc).strip()[:400],
        "meta_keywords": _WS_RE.sub(" ", parser.meta_keywords).strip()[:300],
        "meta_rating": _WS_RE.sub(" ", parser.meta_rating).strip()[:100],
        "h1": clean(parser.h1_parts, 300),
        "snippet": clean(parser.body_parts, 800),
    }


def classify_probe_outcome(http_code: int, meta: dict[str, str]) -> str:
    combined = " ".join([meta["title"], meta["meta_desc"], meta["h1"], meta["snippet"]])
    if _CHALLENGE_RE.search(combined):
        return "challenge"
    if http_code in (403, 429, 503):
        return "challenge"
    if http_code in (404, 410) or http_code >= 500:
        return "dead"
    if _PARKED_RE.search(combined):
        return "parked"
    return "ok"


async def _fetch_with_redirects(
    session: aiohttp.ClientSession,
    start_url: str,
) -> tuple[int, str, str, bytes, str | None]:
    """Fetch URL with manual redirect validation (http/https only, max 3 hops, 64 KB cap)."""
    url = start_url
    for _ in range(MAX_REDIRECTS + 1):
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise SSRFBlockedError(f"Unsupported redirect target: {url}")

        # aiohttp bypasses custom resolvers for IP literals, so check them explicitly.
        try:
            ipaddress.ip_address(parsed.hostname)
            if not is_public_ip(parsed.hostname):
                raise SSRFBlockedError(f"Non-public IP literal: {parsed.hostname}")
        except ValueError:
            pass

        async with session.get(url, allow_redirects=False) as resp:
            if resp.status in (301, 302, 303, 307, 308):
                loc = resp.headers.get("Location")
                if loc:
                    url = urljoin(url, loc)
                    continue

            ctype = (resp.headers.get("Content-Type") or "").lower()
            if ctype and "html" not in ctype and "xhtml" not in ctype:
                return resp.status, (parsed.hostname or "").lower(), ctype, b"", None

            buf = bytearray()
            async for chunk in resp.content.iter_chunked(CHUNK_BYTES):
                buf.extend(chunk)
                if len(buf) >= MAX_BODY_BYTES:
                    break

            return (
                resp.status,
                (parsed.hostname or "").lower(),
                ctype,
                bytes(buf[:MAX_BODY_BYTES]),
                resp.charset,
            )

    return 310, (urlsplit(url).hostname or "").lower(), "", b"", None


async def _probe_one(
    session: aiohttp.ClientSession, domain: str, now: int
) -> PageProbe:
    last_code = 0
    last_host = domain

    for scheme in ("https", "http"):
        try:
            code, final_host, ctype, body, charset = await _fetch_with_redirects(
                session, f"{scheme}://{domain}/"
            )
            last_code = code
            last_host = final_host or domain

            if ctype and "html" not in ctype and "xhtml" not in ctype:
                return PageProbe(
                    domain=domain,
                    status="non_html",
                    http_code=code,
                    final_host=last_host,
                    checked_at=now,
                )

            meta = parse_html_metadata(body, charset)
            status = classify_probe_outcome(code, meta)
            return PageProbe(
                domain=domain,
                status=status,
                http_code=code,
                final_host=last_host,
                title=meta["title"],
                meta_desc=meta["meta_desc"],
                meta_keywords=meta["meta_keywords"],
                meta_rating=meta["meta_rating"],
                h1=meta["h1"],
                snippet=meta["snippet"],
                checked_at=now,
            )
        except SSRFBlockedError:
            return PageProbe(domain=domain, status="blocked_ssrf", checked_at=now)
        except (aiohttp.ClientError, TimeoutError, OSError, UnicodeError) as exc:
            log.debug("Probe %s://%s failed: %s", scheme, domain, exc)
            continue

    return PageProbe(
        domain=domain,
        status="dead",
        http_code=last_code,
        final_host=last_host,
        checked_at=now,
    )


def load_probe_cache(path: Path | None) -> dict[str, PageProbe]:
    if not path or not path.exists():
        return {}
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        return {d: PageProbe(**rec) for d, rec in raw.items()}
    except (OSError, json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
        log.warning("Failed to load probe cache %s: %s", path, exc)
        return {}


def save_probe_cache(path: Path | None, cache: dict[str, PageProbe]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({d: asdict(p) for d, p in cache.items()}, f, ensure_ascii=False)


def _is_cache_fresh(probe: PageProbe, now: int) -> bool:
    ttl = (
        CACHE_TTL_DEAD_SEC
        if probe.status in ("dead", "blocked_ssrf")
        else CACHE_TTL_OK_SEC
    )
    return (now - probe.checked_at) < ttl


async def _probe_batch_async(
    domains: list[str],
    concurrency: int = 80,
) -> dict[str, PageProbe]:
    now = int(time.time())
    sem = asyncio.Semaphore(concurrency)
    resolver = _SafeResolver()
    connector = aiohttp.TCPConnector(
        limit=concurrency,
        resolver=resolver,
        ssl=False,  # Some legacy/mirror sites have mismatched certs; we only read public HTML.
        enable_cleanup_closed=True,
    )
    timeout = aiohttp.ClientTimeout(total=TOTAL_TIMEOUT, sock_connect=CONNECT_TIMEOUT)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
        "Accept-Language": "ru,en;q=0.8",
    }

    async def bound_probe(sess: aiohttp.ClientSession, d: str) -> PageProbe:
        async with sem:
            return await _probe_one(sess, d, now)

    results: dict[str, PageProbe] = {}
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers=headers,
        cookie_jar=aiohttp.DummyCookieJar(),
    ) as session:
        tasks = [asyncio.create_task(bound_probe(session, d)) for d in domains]
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results[res.domain] = res

    return results


def probe_domains(
    domains: list[str],
    cache_path: Path | None = None,
    concurrency: int = 80,
) -> dict[str, PageProbe]:
    """Probe domains for HTML metadata using cache for fresh entries."""
    now = int(time.time())
    cache = load_probe_cache(cache_path)

    results: dict[str, PageProbe] = {}
    to_fetch: list[str] = []
    for d in domains:
        cached = cache.get(d)
        if cached and _is_cache_fresh(cached, now):
            results[d] = cached
        else:
            to_fetch.append(d)

    log.info("HTML probe: %d cached, %d to fetch", len(results), len(to_fetch))
    if to_fetch:
        fetched = asyncio.run(_probe_batch_async(to_fetch, concurrency=concurrency))
        results.update(fetched)
        cache.update(fetched)
        save_probe_cache(cache_path, cache)

    return results
