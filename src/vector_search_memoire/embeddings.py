"""Embedding providers.

The preferred provider is sentence-transformers. The fallback provider is a
deterministic semantic hashing model designed for the synthetic corpus. It makes
the project runnable offline and also gives a pedagogical bridge from sparse
lexical vectors to dense semantic vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util

import numpy as np

from .dataset import SEMANTIC_GROUPS, TOPICS
from .text import STOPWORDS, tokenize


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


class BaseEmbedder:
    name = "base"

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


def _stable_seed(value: str, random_state: int) -> int:
    digest = hashlib.sha256(f"{random_state}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)


def _unit_vector(key: str, dim: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(_stable_seed(key, random_state))
    vector = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    return vector


@dataclass
class SemanticHashingEmbedder(BaseEmbedder):
    """Offline dense embedder with controlled synonym behavior."""

    dim: int = 128
    random_state: int = 42
    lexical_weight: float = 0.35
    semantic_weight: float = 1.8
    name: str = "semantic-hashing-fallback"

    def __post_init__(self) -> None:
        self.semantic_vectors = {
            concept: _unit_vector(f"concept:{concept}", self.dim, self.random_state)
            for concept in SEMANTIC_GROUPS
        }
        self.topic_vectors = {
            topic: _unit_vector(f"topic:{topic}", self.dim, self.random_state)
            for topic in TOPICS
        }
        self.token_to_concepts = self._build_token_to_concepts()
        self.token_to_topics = self._build_token_to_topics()

    def _build_token_to_concepts(self) -> dict[str, set[str]]:
        mapping: dict[str, set[str]] = {}
        for concept, phrases in SEMANTIC_GROUPS.items():
            for phrase in phrases + [concept.replace("_", " ")]:
                for token in tokenize(phrase):
                    if token in STOPWORDS:
                        continue
                    mapping.setdefault(token, set()).add(concept)
        return mapping

    def _build_token_to_topics(self) -> dict[str, set[str]]:
        mapping: dict[str, set[str]] = {}
        for topic, spec in TOPICS.items():
            terms = list(spec["domain_terms"]) + [str(spec["label"])]  # type: ignore[arg-type]
            for term in terms:
                for token in tokenize(term):
                    mapping.setdefault(token, set()).add(topic)
        return mapping

    def _encode_one(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)

        vector = np.zeros(self.dim, dtype=np.float32)
        total_weight = 0.0
        for token in tokens:
            vector += self.lexical_weight * _unit_vector(f"token:{token}", self.dim, self.random_state)
            total_weight += self.lexical_weight

            for concept in self.token_to_concepts.get(token, ()):
                vector += self.semantic_weight * self.semantic_vectors[concept]
                total_weight += self.semantic_weight

            for topic in self.token_to_topics.get(token, ()):
                vector += 0.8 * self.topic_vectors[topic]
                total_weight += 0.8

        if total_weight:
            vector /= total_weight
        return vector.astype(np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.vstack([self._encode_one(text) for text in texts]).astype(np.float32)
        return l2_normalize(matrix)


@dataclass
class SentenceTransformerEmbedder(BaseEmbedder):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)


def sentence_transformers_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def build_embedder(
    provider: str = "auto",
    dim: int = 128,
    random_state: int = 42,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> BaseEmbedder:
    """Create an embedder.

    provider values:
    - auto: sentence-transformers if installed, fallback otherwise
    - sentence-transformers: require SentenceTransformer
    - fallback: deterministic semantic hashing
    """

    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name=model_name)
    if provider == "fallback":
        return SemanticHashingEmbedder(dim=dim, random_state=random_state)
    if provider != "auto":
        raise ValueError(f"Unknown embedding provider: {provider}")

    if sentence_transformers_available():
        try:
            return SentenceTransformerEmbedder(model_name=model_name)
        except Exception:
            return SemanticHashingEmbedder(dim=dim, random_state=random_state)
    return SemanticHashingEmbedder(dim=dim, random_state=random_state)
