"""Deterministic domain filters, DGA/leetspeak detection, and similarity tainting."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from .normalize import PSL, DomainMatcher

SKETCHY_TLDS = frozenset(
    {
        "top",
        "click",
        "cfd",
        "sbs",
        "buzz",
        "win",
        "fun",
        "vip",
        "site",
        "online",
        "club",
        "art",
        "lat",
        "movie",
        "fan",
        "cam",
        "icu",
        "monster",
        "cyou",
        "quest",
        "rest",
        "lol",
        "pics",
        "bid",
        "loan",
        "gdn",
        "work",
        "casa",
        "xyz",
        "biz",
        "vu",
        "pw",
        "shop",
        "store",
        "live",
        "pro",
        "porn",
        "sex",
        "xxx",
        "adult",
        "gay",
        "webcam",
        "bet",
        "casino",
        "poker",
        "bond",
        "mom",
        "love",
        "trade",
        "party",
        "date",
        "men",
        "fit",
    }
)

ADULT_TLDS = frozenset({"porn", "sex", "xxx", "adult", "cam", "webcam", "gay"})
GAMBLING_TLDS = frozenset({"casino", "bet", "poker"})

_RANDOM_SUFFIX_RE = re.compile(r"-(?:[a-z0-9]{2,6})$")
_DIGITS_RE = re.compile(r"\d+")
_LEETSPEAK_TOKEN_RE = re.compile(r"[a-z]+\d+[a-z]+")
_CONSONANT_CLUSTER_RE = re.compile(r"[bcdfghjklmnpqrstvwxz]{5,}")

_SAFE_ALNUM_TOKENS = frozenset(
    {
        "mp3",
        "mp4",
        "web3",
        "b2b",
        "b2c",
        "p2p",
        "k8s",
        "w3c",
        "w3schools",
        "v2ray",
        "v2box",
        "h2o",
        "ipv4",
        "ipv6",
        "utf8",
        "base64",
        "x86",
        "x64",
        "sha256",
        "md5",
        "http2",
        "http3",
        "html5",
        "css3",
        "es6",
        "ps4",
        "ps5",
        "4pda",
        "3d",
        "24h",
        "365d",
        "i18n",
        "l10n",
        "a11y",
        "c2c",
        "g2a",
        "f1",
    }
)

_GAMBLING_RE = re.compile(
    r"(?:casino|kazino|azino|vulkan|wulkan|vavada|1win|1xbet|1xlite|mostbet|"
    r"melbet|pin-?up|pinco|admiralx|royalrussia|bezdep|slotozal|playfortuna|"
    r"riobet|joycas|pokerdom|chpokers|fonbet|olimpbet|parimatch|ggbet|bukmek|"
    r"stavk|spinbetter|megapari|linebet|betwinner|888starz|dragonmoney|up-?x|"
    r"gmsdeluxe|azart|jackpot|megaways|clusterpays|cabura|veegastars|vegastars|"
    r"casebattle|clashofslots|slottgames|form-reg-promo-code|betting|"
    r"slots?(?:online|site|club|vip|777|\d))"
)

_PORN_RE = re.compile(
    r"(?:porn|xxx|xvideo|xhamster|xnxx|redtube|youporn|chaturbate|stripchat|"
    r"bongacams|brazzers|hentai|milf|escort|prostit|putan|shlyuh|intim|dosug|"
    r"seks|ebal|ebka|trah|siski|pizda|livejasmin|camsoda|spankbang|tube8|"
    r"anal(?:forum|foto|galore|planet|sex|porn)|bdsm|pussy|dickof|bitchwithdick|"  # codespell:ignore foto
    r"fetish|erotic|eroboom|sexfree|cosmosex|clicsex|forumosexe|gayerotic|"
    r"runetki|teamskeet|naughtymag|boyfriendoncam|r-34|livegore|smolensklust|"
    r"habarovsklust|manga18free|adult(?:time|work|prime|ptv|chat)|"
    r"gay(?:24|s\.|tube|porn|sex|city|comics|tourist)|fap(?:ello|house|club|er))"
)

_PIRACY_KINO_RE = re.compile(
    r"(?:lord-?film|lordserial|lord-?\d|seriallord|hdrezka|rezka|kinogo|kinokrad|"
    r"kinoprofi|kinotochka|kinoknopka|kinobar|kinozo|filmix|baskino|gidonline|"
    r"seasonvar|lostfilm|zetflix|zagonka|dorama(?:tv|on|\.)|animego|anilibria|"
    r"yummyanime|animebesst|fanserials|smotret-?onlain|multfilm)"
)

_PROBIV_LEAKS_RE = re.compile(
    r"(?:probiv|deanon|eyeofgod|sherlok|sherlockk|quickosint|osintkit|"
    r"slivsklad|skladchinakursov|dolinakursov|shopkurs|freeaccount|hq-accounts)"
)

_ILLICIT_RU_RE = re.compile(
    r"(?:medspravk|spravki|propiska|chhastniy-zaym|zaem-|diplom-?kupit|"
    r"onliprava|kraken-|krakenoff|blacksprut|kladomg|hydra-?onion|"
    r"semyanich|konoplisemena|marijuanagrowing|pyramidseeds|hohlandseeds|"
    r"luna-seeds|seedstockers|torogrow|pevgrow|growgoodbud|rus-anabolik|"
    r"vapeskidon|slimsigar|sigarus|onsigaret|tabachkaplus|bigrc|napas420|fenix24rc)"
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
        if len(stem) >= 3:
            counts[stem] += 1
    return counts


def has_mirror_shape(domain: str, psl: PSL) -> bool:
    sld = psl.sld(domain)
    tld = domain.rsplit(".", 1)[-1]
    if tld in SKETCHY_TLDS:
        return True
    if any(ch.isdigit() for ch in sld):
        return True
    return bool(_RANDOM_SUFFIX_RE.search(sld))


def dga_signals(domain: str, psl: PSL) -> dict[str, bool]:
    """Detect DGA / leetspeak / random-character anomalies in the registrable label."""
    sld = psl.sld(domain)
    tokens = [t for t in re.split(r"[-_]", sld) if t]

    leetspeak = any(
        _LEETSPEAK_TOKEN_RE.search(t) and t not in _SAFE_ALNUM_TOKENS for t in tokens
    )
    consonant_cluster = bool(_CONSONANT_CLUSTER_RE.search(sld))

    letters = [ch for ch in sld if ch.isalpha()]
    vowels = sum(ch in "aeiouy" for ch in letters)
    low_vowels = len(letters) >= 8 and (vowels / len(letters)) < 0.15

    return {
        "leetspeak": leetspeak,
        "consonant_cluster": consonant_cluster,
        "low_vowels": low_vowels,
    }


def is_mirror_cluster(
    domain: str,
    psl: PSL,
    stem_counts: Counter[str],
    min_cluster: int = 10,
) -> bool:
    stem = domain_stem(psl.sld(domain))
    if len(stem) < 3:
        return False
    # Require a slightly higher count for very short 3-char stems (e.g. cat, lev, rox).
    threshold = max(min_cluster, 15) if len(stem) == 3 else min_cluster
    if stem_counts[stem] < threshold:
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
    if _PROBIV_LEAKS_RE.search(target):
        return "rule:probiv_leaks"
    if _ILLICIT_RU_RE.search(target):
        return "rule:illicit_ru"
    return None


@dataclass
class TaintIndex:
    """Tracks registrable roots and stems already seen in dropped junk domains."""

    tainted_roots: set[str]
    tainted_stems: Counter[str]

    @classmethod
    def build(
        cls,
        dropped_domains: Iterable[str],
        psl: PSL,
        allow_matcher: DomainMatcher,
    ) -> TaintIndex:
        roots: set[str] = set()
        stems: Counter[str] = Counter()

        for d in dropped_domains:
            if allow_matcher.match(d) or psl.is_public_suffix(d):
                continue
            reg = psl.registrable(d)
            if not allow_matcher.match(reg) and not psl.is_public_suffix(reg):
                roots.add(reg)
            stem = domain_stem(psl.sld(d))
            if len(stem) >= 3:
                stems[stem] += 1

        return cls(tainted_roots=roots, tainted_stems=stems)

    def root_is_tainted(self, domain: str, psl: PSL) -> bool:
        return psl.registrable(domain) in self.tainted_roots

    def stem_drop_count(self, domain: str, psl: PSL) -> int:
        stem = domain_stem(psl.sld(domain))
        if len(stem) < 3:
            return 0
        return self.tainted_stems.get(stem, 0)


def apply_deterministic_filters(
    domains: list[str],
    psl: PSL,
    feed_matcher: DomainMatcher,
    allow_matcher: DomainMatcher,
    stem_counts: Counter[str],
) -> tuple[list[str], dict[str, list[str]]]:
    """Filter domains using allowlist, feeds, mirror clusters, and strict RU junk rules."""
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
