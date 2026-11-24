"""Deterministic domain filters: threat feeds, mirror clusters, and high-precision RU junk rules."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from .normalize import DomainMatcher, PSL

SKETCHY_TLDS = frozenset({
    "top", "click", "cfd", "sbs", "buzz", "win", "fun", "vip", "site",
    "online", "club", "art", "lat", "movie", "fan", "cam", "icu", "monster",
    "cyou", "quest", "rest", "lol", "pics", "bid", "loan", "gdn", "work",
    "casa", "xyz", "biz", "vu", "pw", "shop", "store", "live", "pro",
    "porn", "sex", "xxx", "adult", "gay", "webcam", "bet", "casino", "poker",
})

ADULT_TLDS = frozenset({"porn", "sex", "xxx", "adult", "cam", "webcam", "gay"})
GAMBLING_TLDS = frozenset({"casino", "bet", "poker"})

_RANDOM_SUFFIX_RE = re.compile(r"-(?:[a-z0-9]{2,6})$")
_DIGITS_RE = re.compile(r"\d+")

_GAMBLING_RE = re.compile(
    r"(?:casino|kazino|azino|vulkan|wulkan|vavada|1win|1xbet|1xlite|mostbet|"
    r"melbet|pin-?up|admiralx|royalrussia|bezdep|slotozal|playfortuna|riobet|"
    r"joycas|pokerdom|fonbet|olimpbet|parimatch|ggbet|bukmek|stavk|spinbetter|"
    r"megapari|linebet|betwinner|888starz|dragonmoney|up-?x|gmsdeluxe|azart|"
    r"jackpot|slots?(?:online|site|club|vip|777|\d)|betting)"
)

_PORN_RE = re.compile(
    r"(?:porn|xxx|xvideo|xhamster|xnxx|redtube|youporn|chaturbate|stripchat|"
    r"bongacams|brazzers|hentai|milf|escort|prostit|putan|shlyuh|intim|dosug|"
    r"seks|ebal|ebka|trah|siski|pizda|livejasmin|camsoda|spankbang|tube8|"
    r"adult(?:time|work|prime|ptv|chat)|gay(?:24|s\.|tube|porn|sex))"
)

_PIRACY_KINO_RE = re.compile(
    r"(?:lordfilm|lordserial|lord-?\d|seriallord|hdrezka|rezka|kinogo|kinokrad|"
    r"kinoprofi|kinotochka|kinoknopka|kinozo|filmix|baskino|gidonline|"
    r"seasonvar|zetflix|zagonka|doramatv|animego|anilibria|yummyanime|"
    r"animebesst|fanserials|smotret-?onlain|multfilm)"
)

_ILLICIT_RU_RE = re.compile(
    r"(?:medspravk|spravki|propiska|chhastniy-zaym|zaem-|diplom-?kupit|"
    r"kraken-|blacksprut|kladomg|hydra-?onion|semena-?konopli|growgoodbud)"
)


def domain_stem(sld: str) -> str:
    """Strip digits and random short hyphenated tails to group mirror domains."""
    stem = _RANDOM_SUFFIX_RE.sub("", sld)
    stem = _DIGITS_RE.sub("", stem)
    return stem.strip("-_")


def build_stem_counts(domains: Iterable[str], psl: PSL) -> Counter[str]:
    counts: Counter[str] = Counter()
    for d in domains:
        stem = domain_stem(psl.sld(d))
        if len(stem) >= 4:
            counts[stem] += 1
    return counts


def has_mirror_shape(domain: str, psl: PSL) -> bool:
    sld = psl.sld(domain)
    tld = domain.rsplit(".", 1)[-1]
    if tld in SKETCHY_TLDS:
        return True
    if any(ch.isdigit() for ch in sld):
        return True
    if _RANDOM_SUFFIX_RE.search(sld):
        return True
    return False


def is_mirror_cluster(
    domain: str,
    psl: PSL,
    stem_counts: Counter[str],
    min_cluster: int = 10,
) -> bool:
    stem = domain_stem(psl.sld(domain))
    if len(stem) < 4 or stem_counts[stem] < min_cluster:
        return False
    return has_mirror_shape(domain, psl)


def match_junk_domain_rule(domain: str, psl: PSL) -> str | None:
    tld = domain.rsplit(".", 1)[-1]
    if tld in ADULT_TLDS:
        return "rule:adult_tld"
    if tld in GAMBLING_TLDS:
        return "rule:gambling_tld"

    target = domain.rsplit(".", 1)[0]
    if _GAMBLING_RE.search(target):
        return "rule:gambling"
    if _PORN_RE.search(target):
        return "rule:porn"
    if _PIRACY_KINO_RE.search(target):
        return "rule:piracy_kino"
    if _ILLICIT_RU_RE.search(target):
        return "rule:illicit_ru"
    return None


def apply_deterministic_filters(
    domains: list[str],
    psl: PSL,
    feed_matcher: DomainMatcher,
    allow_matcher: DomainMatcher,
    stem_counts: Counter[str],
) -> tuple[list[str], dict[str, list[str]]]:
    """Filter domains using allowlist, feeds, mirror clusters, and RU junk rules."""
    kept: list[str] = []
    dropped: dict[str, list[str]] = {}

    for d in domains:
        if allow_matcher.match(d):
            kept.append(d)
            continue

        if feed_matcher.match(d):
            dropped.setdefault("feed", []).append(d)
            continue

        rule = match_junk_domain_rule(d, psl)
        if rule is not None:
            dropped.setdefault(rule, []).append(d)
            continue

        if is_mirror_cluster(d, psl, stem_counts):
            dropped.setdefault("mirror_cluster", []).append(d)
            continue

        kept.append(d)

    return kept, dropped
