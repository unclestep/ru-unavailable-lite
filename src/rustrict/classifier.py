"""Feature extraction and ML classifiers: Domain+Structural and HTML+Domain content."""

from __future__ import annotations

import logging
import math
import random
import re
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .filters import SKETCHY_TLDS, dga_signals, domain_stem, has_mirror_shape, match_junk_domain_rule
from .normalize import DomainMatcher, PSL
from .probe import PageProbe

log = logging.getLogger(__name__)

_HTML_CATEGORY_SIGNALS: dict[str, tuple[re.Pattern[str], ...]] = {
    "gambling": (
        re.compile(r"\b(?:онлайн[- ]?казино|казино онлайн|игровые автоматы|игровых автоматов)\b", re.I),
        re.compile(r"\b(?:фриспин\w*|бездепозитн\w*|бонус за регистрацию|промокод\w* казино)\b", re.I),
        re.compile(r"\b(?:слоты онлайн|играть в слоты|игровые слоты|крутить слоты|джекпот|megaways)\b", re.I),
        re.compile(r"\b(?:букмекерск\w* контор\w*|ставки на спорт|зеркало бк|рабочее зеркало)\b", re.I),
        re.compile(r"\b(?:online casino|free spins|slot machines|sports betting|live roulette)\b", re.I),
        re.compile(r"\b(?:1xbet|1win|vavada|pin[- ]?up|pinco|azino777|vulkan|mostbet|melbet|cabura)\b", re.I),
    ),
    "porn": (
        re.compile(r"\b(?:порно онлайн|бесплатное порно|порно видео|порно ролики|порнух\w*)\b", re.I),
        re.compile(r"\b(?:секс видео|смотреть порно|домашнее порно|русское порно|хентай)\b", re.I),
        re.compile(r"\b(?:интим услуг\w*|проститутк\w*|индивидуалк\w*|эскорт услуг\w*|вебкам модел\w*)\b", re.I),
        re.compile(r"\b(?:free porn|porn videos|xxx videos|live sex cams?|porn tube|hardcore sex|bdsm)\b", re.I),
    ),
    "piracy_kino": (
        re.compile(r"\b(?:смотреть онлайн бесплатно|фильмы онлайн бесплатно|сериалы онлайн бесплатно)\b", re.I),
        re.compile(r"\b(?:в хорошем качестве|в качестве hd|hd 720|hd 1080|full hd бесплатно)\b", re.I),
        re.compile(r"\b(?:все серии подряд|все сезоны онлайн|смотреть фильм онлайн)\b", re.I),
        re.compile(r"\b(?:lord-?film|hdrezka|kinogo|baskino|seasonvar|lostfilm|animego|zagonka|kinobar)\b", re.I),
    ),
    "probiv_leaks": (
        re.compile(r"\b(?:пробив по номеру|пробить человека|глаз бога|деанон|бот пробива)\b", re.I),
        re.compile(r"\b(?:складчина курсов|слив курсов|купить аккаунты|базы данных пробив)\b", re.I),
    ),
    "illicit_ru": (
        re.compile(r"\b(?:купить справку|медсправк\w* без посещения|купить больничный|купить права)\b", re.I),
        re.compile(r"\b(?:временная регистрация купить|купить прописку|купить диплом)\b", re.I),
        re.compile(r"\b(?:даркнет маркет|купить закладк\w*|моментальные клады|семена конопли)\b", re.I),
        re.compile(r"\b(?:купить сигареты оптом|вейп шоп|доставка алкоголя ночью|купить анаболики)\b", re.I),
    ),
}

_ADULT_META_RE = re.compile(r"(?:adult|mature|rta-5042)", re.I)


def html_keyword_hits(probe: PageProbe) -> tuple[str | None, int]:
    """Return the strongest matching junk HTML category and its distinct signal count."""
    text = probe.combined_text()
    if not text:
        return None, 0

    best_cat: str | None = None
    best_hits = 0
    for cat, patterns in _HTML_CATEGORY_SIGNALS.items():
        hits = sum(1 for rx in patterns if rx.search(text))
        if hits > best_hits:
            best_cat = cat
            best_hits = hits
    return best_cat, best_hits


def match_hard_html_junk(
    probe: PageProbe,
    psl: PSL,
    feed_matcher: DomainMatcher,
) -> str | None:
    """Return hard-drop reason when HTML/redirect evidence is unambiguous."""
    if probe.status == "parked":
        return "html:parked"
    if probe.status == "blocked_ssrf":
        return "html:ssrf"

    if probe.final_host and probe.final_host != probe.domain:
        if feed_matcher.match(probe.final_host):
            return "html:redirect_feed"
        rule = match_junk_domain_rule(probe.final_host, psl)
        if rule:
            return f"html:redirect_{rule}"

    if probe.meta_rating and _ADULT_META_RE.search(probe.meta_rating):
        return "html:meta_adult"

    return None


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _structural_features(
    domains: list[str],
    psl: PSL,
    stem_counts: Counter[str],
) -> np.ndarray:
    rows = []
    for d in domains:
        sld = psl.sld(d)
        tld = d.rsplit(".", 1)[-1]
        n = max(len(sld), 1)
        digits = sum(ch.isdigit() for ch in sld)
        hyphens = sld.count("-")
        stem = domain_stem(sld)
        stem_freq = math.log1p(stem_counts.get(stem, 0)) if len(stem) >= 3 else 0.0
        dga = dga_signals(d, psl)
        rows.append([
            float(len(sld)),
            digits / n,
            float(hyphens),
            _shannon_entropy(sld),
            1.0 if tld in SKETCHY_TLDS else 0.0,
            1.0 if has_mirror_shape(d, psl) else 0.0,
            1.0 if dga["leetspeak"] else 0.0,
            1.0 if dga["consonant_cluster"] else 0.0,
            stem_freq,
        ])
    return np.asarray(rows, dtype=np.float64)


def _threshold_for_precision(y_true: np.ndarray, scores: np.ndarray, target_precision: float) -> float:
    precision, _, thresholds = precision_recall_curve(y_true, scores)
    ok = np.where(precision[:-1] >= target_precision)[0]
    return float(thresholds[ok[0]]) if len(ok) else 0.99


class DomainClassifier:
    """Domain-name + structural classifier (lexical char n-grams + DGA/TLD/stem features)."""

    def __init__(self, psl: PSL, stem_counts: Counter[str]) -> None:
        self.psl = psl
        self.stem_counts = stem_counts
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=3,
            sublinear_tf=True,
            max_features=200_000,
        )
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1500, C=1.0, class_weight="balanced")
        self.strict_threshold = 0.95

    def _transform(self, domains: list[str], fit: bool = False):
        x_text = self.vectorizer.fit_transform(domains) if fit else self.vectorizer.transform(domains)
        struct = _structural_features(domains, self.psl, self.stem_counts)
        x_struct = self.scaler.fit_transform(struct) if fit else self.scaler.transform(struct)
        return hstack([x_text, csr_matrix(x_struct)], format="csr")

    def fit(self, junk_domains: list[str], legit_domains: list[str], min_precision: float = 0.99) -> float:
        rng = random.Random(42)
        cap = 80_000
        junk = rng.sample(junk_domains, cap) if len(junk_domains) > cap else list(junk_domains)
        legit = rng.sample(legit_domains, cap) if len(legit_domains) > cap else list(legit_domains)

        X = junk + legit
        y = np.array([1] * len(junk) + [0] * len(legit))
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.model.fit(self._transform(X_tr, fit=True), y_tr)
        scores = self.model.predict_proba(self._transform(X_te))[:, 1]
        auc = float(roc_auc_score(y_te, scores))
        self.strict_threshold = max(_threshold_for_precision(y_te, scores, min_precision), 0.85)
        log.info(
            "DomainClassifier trained: junk=%d legit=%d ROC-AUC=%.4f strict_thr=%.3f",
            len(junk),
            len(legit),
            auc,
            self.strict_threshold,
        )
        return auc

    def predict_scores(self, domains: list[str]) -> np.ndarray:
        if not domains:
            return np.array([], dtype=np.float64)
        return self.model.predict_proba(self._transform(domains))[:, 1]


class ContentClassifier:
    """HTML + Domain TF-IDF classifier trained via distant supervision on probed pages."""

    def __init__(self) -> None:
        self.word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=80_000,
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            sublinear_tf=True,
            max_features=80_000,
        )
        self.model = LogisticRegression(max_iter=1500, C=1.0, class_weight="balanced")
        self.threshold = 0.85
        self.fitted = False

    @staticmethod
    def _doc_text(probe: PageProbe) -> str:
        return f"{probe.domain} {probe.final_host} {probe.combined_text()}".strip()

    def _transform(self, probes: list[PageProbe], fit: bool = False):
        docs = [self._doc_text(p) for p in probes]
        domains_titles = [f"{p.domain} {p.title}".strip() for p in probes]
        if fit:
            xw = self.word_vec.fit_transform(docs)
            xc = self.char_vec.fit_transform(domains_titles)
        else:
            xw = self.word_vec.transform(docs)
            xc = self.char_vec.transform(domains_titles)
        return hstack([xw, xc], format="csr")

    def fit(self, junk_probes: list[PageProbe], legit_probes: list[PageProbe], min_precision: float = 0.97) -> bool:
        if len(junk_probes) < 40 or len(legit_probes) < 40:
            log.info(
                "ContentClassifier skipped: insufficient labeled pages (junk=%d, legit=%d)",
                len(junk_probes),
                len(legit_probes),
            )
            return False

        X = list(junk_probes) + list(legit_probes)
        y = np.array([1] * len(junk_probes) + [0] * len(legit_probes))
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.model.fit(self._transform(X_tr, fit=True), y_tr)
        scores = self.model.predict_proba(self._transform(X_te))[:, 1]
        auc = float(roc_auc_score(y_te, scores))
        self.threshold = max(_threshold_for_precision(y_te, scores, min_precision), 0.75)
        self.fitted = True
        log.info(
            "ContentClassifier trained: junk=%d legit=%d ROC-AUC=%.4f thr=%.3f",
            len(junk_probes),
            len(legit_probes),
            auc,
            self.threshold,
        )
        return True

    def predict_scores(self, probes: list[PageProbe]) -> np.ndarray:
        if not probes or not self.fitted:
            return np.zeros(len(probes), dtype=np.float64)
        return self.model.predict_proba(self._transform(probes))[:, 1]
