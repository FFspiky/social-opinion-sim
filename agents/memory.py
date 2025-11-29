# agents/memory.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from datetime import datetime, timedelta

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MemoryItem:
    text: str
    created_at: datetime
    importance: float
    embedding: np.ndarray = field(repr=False)


class MemoryStream:
    """
    精简版 Generative Agents 记忆流：
    - 全局共享一个 SentenceTransformer 模型，避免重复加载
    - 支持重要度、时效性加权检索
    """

    _embed_model = None  # 共享嵌入模型

    def __init__(
        self,
        llm_client,
        reflection_threshold: float = 30.0,
        recency_half_life_hours: float = 6.0,
    ):
        self.llm_client = llm_client

        if MemoryStream._embed_model is None:
            MemoryStream._embed_model = SentenceTransformer("models/all-MiniLM-L6-v2")
        self.model = MemoryStream._embed_model

        self.memories: List[MemoryItem] = []
        self.reflection_threshold = reflection_threshold
        self.recency_half_life_hours = recency_half_life_hours

    # ---- 基础操作 ----

    def add(self, text: str):
        emb = self.model.encode([text])[0]
        item = MemoryItem(
            text=text,
            created_at=datetime.now(),
            importance=5.0,  # 默认重要度，可在 _score_importance 中评估
            embedding=emb,
        )
        self.memories.append(item)

    def retrieve(self, query: str, k: int = 5) -> List[MemoryItem]:
        if not self.memories:
            return []
        query_emb = self.model.encode([query])[0]
        scores = []
        for m in self.memories:
            sim = cosine_similarity([query_emb], [m.embedding])[0][0]
            recency = self._recency_weight(m.created_at)
            score = sim * 0.7 + recency * 0.3
            scores.append(score)
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.memories[i] for i in top_indices]

    # ---- 反思与重要度 ----

    def maybe_reflect(self):
        """
        简化的反思：当记忆数量或重要度累积超过阈值时触发，示例中仅重置计数。
        """
        if len(self.memories) >= self.reflection_threshold:
            # 这里可以接入 llm_client 做更复杂的总结/抽象
            self.memories = self.memories[-self.reflection_threshold :]

    def _recency_weight(self, created_at: datetime) -> float:
        """
        根据时间衰减计算权重（半衰期 recency_half_life_hours）。
        """
        hours = (datetime.now() - created_at).total_seconds() / 3600
        decay = 0.5 ** (hours / self.recency_half_life_hours)
        return decay
