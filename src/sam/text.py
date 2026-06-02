from __future__ import annotations

import hashlib
import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*|[\u4e00-\u9fff]{2,}")

STOPWORDS = {
    "the",
    "and",
    "or",
    "of",
    "in",
    "to",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "for",
    "with",
    "that",
    "which",
    "what",
    "where",
    "when",
    "who",
    "whose",
    "how",
    "by",
    "on",
    "as",
    "at",
    "from",
    "this",
    "it",
    "its",
    "he",
    "she",
    "him",
    "her",
    "his",
    "hers",
    "you",
    "your",
    "i",
    "me",
    "my",
    "we",
    "our",
    "they",
    "them",
    "their",
    "had",
    "has",
    "have",
    "do",
    "does",
    "did",
    "so",
    "but",
    "not",
    "no",
    "yes",
    "one",
    "be",
}


def tokenize(text: str) -> list[str]:
    """把中英文文本切成稳定 token，用于本地 embedding 和关键词抽取。"""

    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def extract_keywords(text: str, limit: int = 8) -> list[str]:
    """用词频抽取轻量关键词，避免第一版依赖复杂 NLP 工具。"""

    counts = Counter(tokenize(text))
    return [token for token, _ in counts.most_common(limit)]


def stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
