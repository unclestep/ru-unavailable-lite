"""Protobuf encoders/decoders for geosite.dat/geoip.dat and split-rules.json exporter."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
from pathlib import Path
from typing import Iterable

from .normalize import PSL, dedup_by_coverage

log = logging.getLogger(__name__)

TYPE_DOMAIN = 2  # Protobuf Domain.Type = Domain (matches domain and subdomains)

GEOIP_DENY = frozenset({"CLOUDFLARE", "CLOUDFRONT", "FASTLY", "AKAMAI", "GOOGLE"})
GEOIP_REQUIRED_FOREIGN = ("PRIVATE", "TELEGRAM", "NETFLIX", "FACEBOOK", "TWITTER")


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | 0x80 if value else byte)
        if not value:
            return bytes(out)


def _tag(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _len_delim(field: int, payload: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(payload)) + payload


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _iter_fields(buf: bytes):
    pos = 0
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _read_varint(buf, pos)
            yield field, value
        elif wire == 2:
            length, pos = _read_varint(buf, pos)
            yield field, buf[pos : pos + length]
            pos += length
        else:
            raise ValueError(f"Unsupported protobuf wire type: {wire}")


def encode_geosite(lists: dict[str, list[str]]) -> bytes:
    out = bytearray()
    for name, domains in lists.items():
        entry = _len_delim(1, name.upper().encode())
        for domain in domains:
            rule = _tag(1, 0) + _varint(TYPE_DOMAIN) + _len_delim(2, domain.encode())
            entry += _len_delim(2, rule)
        out += _len_delim(1, entry)
    return bytes(out)


def decode_geosite(raw: bytes) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for field, payload in _iter_fields(raw):
        if field != 1:
            continue
        name = None
        domains: list[str] = []
        for sub_field, sub_payload in _iter_fields(payload):
            if sub_field == 1:
                name = sub_payload.decode()
            elif sub_field == 2:
                val = None
                for dom_field, dom_payload in _iter_fields(sub_payload):
                    if dom_field == 2:
                        val = dom_payload.decode()
                if val:
                    domains.append(val)
        if name:
            result[name] = domains
    return result


def encode_geoip(lists: dict[str, list[tuple[bytes, int]]]) -> bytes:
    out = bytearray()
    for code, networks in lists.items():
        entry = _len_delim(1, code.upper().encode())
        for packed, prefix in networks:
            cidr = _len_delim(1, packed) + _tag(2, 0) + _varint(prefix)
            entry += _len_delim(2, cidr)
        out += _len_delim(1, entry)
    return bytes(out)


def decode_geoip(raw: bytes) -> dict[str, list[tuple[bytes, int]]]:
    result: dict[str, list[tuple[bytes, int]]] = {}
    for field, payload in _iter_fields(raw):
        if field != 1:
            continue
        code = None
        networks: list[tuple[bytes, int]] = []
        for sub_field, sub_payload in _iter_fields(payload):
            if sub_field == 1:
                code = sub_payload.decode()
            elif sub_field == 2:
                packed = prefix = None
                for c_field, c_payload in _iter_fields(sub_payload):
                    if c_field == 1:
                        packed = c_payload
                    elif c_field == 2:
                        prefix = c_payload
                if packed is not None and prefix is not None:
                    networks.append((packed, prefix))
        if code:
            result[code] = networks
    return result


def collapse_packed_cidrs(pairs: Iterable[tuple[bytes, int]]) -> list[tuple[bytes, int]]:
    v4, v6 = [], []
    for packed, prefix in pairs:
        net = ipaddress.ip_network((packed, prefix), strict=False)
        if net.version == 4:
            v4.append(net)
        else:
            v6.append(net)

    out = [(n.network_address.packed, n.prefixlen) for n in ipaddress.collapse_addresses(v4)]
    out.extend((n.network_address.packed, n.prefixlen) for n in ipaddress.collapse_addresses(v6))
    return out


def cidr_str(packed: bytes, prefix: int) -> str:
    return f"{ipaddress.ip_address(packed)}/{prefix}"


def build_geoip_lists(
    foreign_geoip: dict[str, list[tuple[bytes, int]]],
    google_first_party: list[tuple[bytes, int]],
    all_domains: Iterable[str],
    psl: PSL,
) -> dict[str, list[tuple[bytes, int]]]:
    """Assemble compact GeoIP categories without CDNs or GCP customer ranges."""
    service_cats = {c: n for c, n in foreign_geoip.items() if len(c) > 2}
    by_lower = {c.lower(): c for c in service_cats}

    selected: dict[str, list[tuple[bytes, int]]] = {}
    for d in all_domains:
        code = by_lower.get(psl.sld(d))
        if code and code not in GEOIP_DENY and code not in selected:
            selected[code] = service_cats[code]

    missing = [code for code in GEOIP_REQUIRED_FOREIGN if not foreign_geoip.get(code)]
    if missing:
        raise ValueError(f"Foreign geoip.dat missing required categories: {', '.join(missing)}")
    for code in GEOIP_REQUIRED_FOREIGN:
        selected[code] = foreign_geoip[code]

    if not google_first_party:
        raise ValueError("Google first-party CIDR list is empty")
    selected["GOOGLE"] = google_first_party

    return {code: collapse_packed_cidrs(selected[code]) for code in sorted(selected)}


def build_split_rules(
    version: str,
    ru_blocked: list[str],
    ru_forbidden: list[str],
    geoip_lists: dict[str, list[tuple[bytes, int]]],
) -> dict:
    """Build ready-to-embed Xray split rules (no external .dat dependencies on clients)."""
    merged_domains = dedup_by_coverage([*ru_blocked, *ru_forbidden])
    xray_domains = [f"domain:{d}" for d in merged_domains]

    proxy_cidrs: list[tuple[bytes, int]] = []
    for code, nets in geoip_lists.items():
        if code == "PRIVATE":
            continue
        proxy_cidrs.extend(nets)
    collapsed_ips = [cidr_str(p, m) for p, m in collapse_packed_cidrs(proxy_cidrs)]

    return {
        "version": version,
        "domains": xray_domains,
        "ips": collapsed_ips,
    }


def write_artifacts(
    out_dir: Path,
    version: str,
    ru_blocked: list[str],
    ru_forbidden: list[str],
    geoip_lists: dict[str, list[tuple[bytes, int]]],
    stats: dict,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    geosite_bytes = encode_geosite({"ru-blocked": ru_blocked, "ru-forbidden": ru_forbidden})
    decoded_gs = decode_geosite(geosite_bytes)
    assert decoded_gs == {"RU-BLOCKED": ru_blocked, "RU-FORBIDDEN": ru_forbidden}

    geoip_bytes = encode_geoip(geoip_lists)
    decoded_gi = decode_geoip(geoip_bytes)
    assert decoded_gi == geoip_lists

    split_rules = build_split_rules(version, ru_blocked, ru_forbidden, geoip_lists)

    files = {
        "geosite.dat": geosite_bytes,
        "geoip.dat": geoip_bytes,
        "split-rules.json": (json.dumps(split_rules, indent=2) + "\n").encode("utf-8"),
        "ru-blocked.txt": ("\n".join(ru_blocked) + "\n").encode("utf-8"),
        "ru-forbidden.txt": ("\n".join(ru_forbidden) + "\n").encode("utf-8"),
    }

    written: dict[str, Path] = {}
    checksum_lines: list[str] = []
    for name in sorted(files):
        p = out_dir / name
        p.write_bytes(files[name])
        digest = hashlib.sha256(files[name]).hexdigest()
        checksum_lines.append(f"{digest}  {name}")
        written[name] = p

    checksum_path = out_dir / "checksums.sha256sum"
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    written["checksums.sha256sum"] = checksum_path

    stats_path = out_dir / "stats.json"
    stats_out = {
        **stats,
        "version": version,
        "output": {
            "ru_blocked": len(ru_blocked),
            "ru_forbidden": len(ru_forbidden),
            "split_domains": len(split_rules["domains"]),
            "split_ips": len(split_rules["ips"]),
            "geoip_categories": {k: len(v) for k, v in geoip_lists.items()},
        },
    }
    stats_path.write_text(json.dumps(stats_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written["stats.json"] = stats_path

    log.info(
        "Exported: ru-blocked=%d, ru-forbidden=%d, split-domains=%d, split-ips=%d",
        len(ru_blocked),
        len(ru_forbidden),
        len(split_rules["domains"]),
        len(split_rules["ips"]),
    )
    return written
