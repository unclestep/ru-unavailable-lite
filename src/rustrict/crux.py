"""Chrome User Experience Report (CrUX) popularity filter via BigQuery."""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .normalize import PSL

log = logging.getLogger(__name__)

CRUX_TABLE = "chrome-ux-report.experimental.country"
DEFAULT_RANK_MAX = 1_000_000


@dataclass
class CruxSnapshot:
    yyyymm: int
    host_ranks: dict[str, int]
    reg_ranks: dict[str, int]


def fetch_crux_snapshot(
    project_id: str,
    psl: PSL,
    rank_max: int = DEFAULT_RANK_MAX,
) -> CruxSnapshot:
    """Query the latest RU CrUX country snapshot from BigQuery without pandas."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)

    probe_sql = f"SELECT MAX(yyyymm) AS m FROM `{CRUX_TABLE}` WHERE yyyymm >= 202401"
    yyyymm = int(next(iter(client.query(probe_sql).result()))["m"])
    log.info("Latest CrUX month: %d", yyyymm)

    query = f"""
        SELECT DISTINCT origin, experimental.popularity.rank AS rank
        FROM `{CRUX_TABLE}`
        WHERE yyyymm = {yyyymm}
          AND country_code = "ru"
          AND experimental.popularity.rank <= {rank_max}
    """

    host_ranks: dict[str, int] = {}
    reg_ranks: dict[str, int] = {}
    total_origins = 0

    for row in client.query(query).result():
        total_origins += 1
        origin = row["origin"]
        rank = int(row["rank"])
        host = (urlsplit(origin).hostname or "").lower().strip(".")
        if host.startswith("www."):
            host = host[4:]
        if not host:
            continue

        if host not in host_ranks or rank < host_ranks[host]:
            host_ranks[host] = rank

        reg = psl.registrable(host)
        if reg not in reg_ranks or rank < reg_ranks[reg]:
            reg_ranks[reg] = rank

    log.info(
        "CrUX RU: %d origins -> %d hosts, %d registrable domains",
        total_origins,
        len(host_ranks),
        len(reg_ranks),
    )
    return CruxSnapshot(yyyymm=yyyymm, host_ranks=host_ranks, reg_ranks=reg_ranks)


def filter_by_crux(domains: list[str], crux: CruxSnapshot, psl: PSL) -> list[str]:
    """Keep domains present in CrUX RU.

    Apex domains match by registrable domain; subdomains must match their exact
    CrUX host so obscure subdomains cannot ride on a popular shared host's rank.
    """
    return list(domain_crux_ranks(domains, crux, psl))


def domain_crux_ranks(domains: list[str], crux: CruxSnapshot, psl: PSL) -> dict[str, int]:
    """Return mapping of domains present in CrUX RU to their popularity rank."""
    out: dict[str, int] = {}
    for d in domains:
        reg = psl.registrable(d)
        if d == reg:
            if reg in crux.reg_ranks:
                out[d] = crux.reg_ranks[reg]
        elif d in crux.host_ranks:
            out[d] = crux.host_ranks[d]
    return out


def save_crux_snapshot(path: Path, snap: CruxSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "yyyymm": snap.yyyymm,
        "host_ranks": snap.host_ranks,
        "reg_ranks": snap.reg_ranks,
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


def load_crux_snapshot(path: Path) -> CruxSnapshot:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return CruxSnapshot(
        yyyymm=int(data["yyyymm"]),
        host_ranks=data["host_ranks"],
        reg_ranks=data["reg_ranks"],
    )
