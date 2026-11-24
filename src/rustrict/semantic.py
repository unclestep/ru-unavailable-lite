"""Multilingual zero-shot semantic HTML classifier powered by fastembed (ONNX Runtime)."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .probe import PageProbe

log = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

JUNK_PROTOTYPES: dict[str, str] = {
    "gambling": (
        "онлайн казино, игровые автоматы, слоты, фриспины, бонус за регистрацию, "
        "ставки на спорт, букмекерская контора, рулетка, джекпот, рабочее зеркало казино, "
        "online casino slots free spins sports betting"
    ),
    "porn": (
        "порно видео онлайн бесплатно, секс ролики, порнуха, эротика, вебкам модели, "
        "индивидуалки, эскорт услуги, проститутки города, хентай, free porn videos xxx adult cams"
    ),
    "piracy_kino": (
        "смотреть фильмы и сериалы онлайн бесплатно в хорошем качестве HD 720 1080, "
        "все серии подряд без регистрации, новинки кино онлайн, пиратский кинотеатр"
    ),
    "probiv_deanon": (
        "пробив человека по номеру телефона и фото, деанон, глаз бога, поиск данных в telegram, "
        "базы данных слив, купить аккаунты, складчина слитых курсов"
    ),
    "illicit_substances": (
        "купить сигареты оптом, вейп шоп, табак доставка, ночная доставка алкоголя 24/7, "
        "семена конопли, каннабис гровинг, анаболики стероиды, купить медсправку и диплом"
    ),
}

LEGIT_PROTOTYPES: dict[str, str] = {
    "news_media": (
        "независимые новости, аналитика, журналистские расследования, политика, экономика, "
        "репортажи, интервью, независимое издание, breaking news journalism media"
    ),
    "tech_dev_ai": (
        "software development, cloud platform, developer documentation, programming, AI tools, "
        "open source repository, API reference, IT infrastructure, cybersecurity"
    ),
    "community_forum": (
        "интернет форум, сообщество, обсуждения, блоги, вопросы и ответы, база знаний, "
        "технический форум, hobby community discussion board"
    ),
    "books_torrents": (
        "электронная библиотека, книги скачать читать, научные статьи, архив файлов, "
        "крупный торрент трекер, digital library books archive"
    ),
    "vpn_privacy": (
        "VPN сервис, обход блокировок, приватность, шифрование, безопасный интернет, "
        "мессенджер, почтовый сервис, privacy proxy encryption"
    ),
}


@dataclass(frozen=True)
class SemanticScore:
    junk_category: str
    junk_sim: float
    legit_category: str
    legit_sim: float

    @property
    def margin(self) -> float:
        return self.junk_sim - self.legit_sim


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


class SemanticClassifier:
    """Zero-shot cosine similarity scorer over multilingual page embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    ) -> None:
        self.model_name = model_name
        self._embed_fn = embed_fn
        self._model = None
        self._junk_names = list(JUNK_PROTOTYPES)
        self._legit_names = list(LEGIT_PROTOTYPES)
        self._junk_vecs: np.ndarray | None = None
        self._legit_vecs: np.ndarray | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._embed_fn is not None:
            return _normalize_rows(np.asarray(self._embed_fn(texts), dtype=np.float32))

        if self._model is None:
            from fastembed import TextEmbedding

            cache_dir = os.environ.get("FASTEMBED_CACHE_PATH") or None
            log.info("Loading ONNX embedding model %s", self.model_name)
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=cache_dir)

        vecs = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        return _normalize_rows(vecs)

    def _ensure_prototypes(self) -> None:
        if self._junk_vecs is None:
            self._junk_vecs = self._embed(
                [JUNK_PROTOTYPES[k] for k in self._junk_names]
            )
        if self._legit_vecs is None:
            self._legit_vecs = self._embed(
                [LEGIT_PROTOTYPES[k] for k in self._legit_names]
            )

    def score_probes(self, probes: Iterable[PageProbe]) -> dict[str, SemanticScore]:
        items = [p for p in probes if p.status == "ok" and len(p.combined_text()) >= 15]
        if not items:
            return {}

        self._ensure_prototypes()
        if self._junk_vecs is None or self._legit_vecs is None:
            raise RuntimeError("Semantic prototype vectors were not initialized")

        docs = [f"{p.domain}: {p.combined_text()[:500]}" for p in items]
        doc_vecs = self._embed(docs)

        junk_sims = doc_vecs @ self._junk_vecs.T
        legit_sims = doc_vecs @ self._legit_vecs.T

        results: dict[str, SemanticScore] = {}
        for i, probe in enumerate(items):
            j_idx = int(np.argmax(junk_sims[i]))
            l_idx = int(np.argmax(legit_sims[i]))
            results[probe.domain] = SemanticScore(
                junk_category=self._junk_names[j_idx],
                junk_sim=float(junk_sims[i, j_idx]),
                legit_category=self._legit_names[l_idx],
                legit_sim=float(legit_sims[i, l_idx]),
            )

        log.info("Semantic zero-shot scored %d HTML pages", len(results))
        return results
