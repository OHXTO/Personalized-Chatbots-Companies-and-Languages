import os
import re
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 读取 docs 目录 → 切块 → TF-IDF 建索引 → query 返回 top_k 段落 + 相似度分数。


@dataclass
class Chunk:
    source: str
    chunk_id: int
    text: str


def _clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _chunk_text(text: str, max_chars: int = 500, overlap: int = 80) -> List[str]:
    """
    Simple character-based chunker.
    Keeps overlap to preserve context across chunk boundaries.
    """
    text = _clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= n:
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


class TfidfRAG:
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.chunks: List[Chunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None  # TF-IDF sparse matrix

    def build(self) -> None:
        if not os.path.isdir(self.docs_dir):
            self.chunks = []
            self.vectorizer = None
            self.matrix = None
            return

        chunks: List[Chunk] = []
        for name in sorted(os.listdir(self.docs_dir)):
            if not name.lower().endswith(".txt"):
                continue
            path = os.path.join(self.docs_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()
            except UnicodeDecodeError:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()

            for i, c in enumerate(_chunk_text(raw)):
                chunks.append(Chunk(source=name, chunk_id=i, text=c))

        self.chunks = chunks
        if not chunks:
            self.vectorizer = None
            self.matrix = None
            return

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english", 
            ngram_range=(1, 2),
            max_features=50000,
        )
        corpus = [c.text for c in chunks]
        self.matrix = self.vectorizer.fit_transform(corpus)


    def query(self, question: str, top_k: int = 4, source_contains: str | None = None):
        if not self.vectorizer or self.matrix is None or not self.chunks:
            return []

        q = question.strip()
        if not q:
            return []

        idx_map = list(range(len(self.chunks)))
        if source_contains:
            s = source_contains.lower()
            idx_map = [i for i, ch in enumerate(self.chunks) if s in ch.source.lower()]
            if not idx_map:
                idx_map = list(range(len(self.chunks)))  # 找不到就退回全量

        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.matrix).flatten()

        ranked = sorted(idx_map, key=lambda i: sims[i], reverse=True)

        results = []
        used = []
        for i in ranked:
            ch = self.chunks[i]

            # 简单去重：同一 source 中 chunk_id 太近的不重复拿, simple deduplication
            too_close = any(
                u.source == ch.source and abs(u.chunk_id - ch.chunk_id) <= 1
                for u in used
            )
            if too_close:
                continue

            results.append((ch, float(sims[i])))
            used.append(ch)

            if len(results) >= top_k:
                break

        return results


