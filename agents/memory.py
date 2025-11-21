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
    简化版 Generative Agents 记忆流：
    - 全局共享一个 SentenceTransformer 模型，避免重复加载
    """

    # 类变量：全局共享
    _embed_model = None

    def __init__(
        self,
        llm_client,
        reflection_threshold: float = 30.0,
        recency_half_life_hours: float = 6.0,
    ):
        self.llm_client = llm_client

        # 只在第一次初始化时加载一次模型
        if MemoryStream._embed_model is None:
            MemoryStream._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = MemoryStream._embed_model

        self.memories = []
        self.reflection_threshold = reflection_threshold
        self.recency_half_life = timedelta(hours=recency_half_life_hours)
        self._importance_accumulator = 0.0

    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]

    def _score_importance(self, text: str) -> float:
        """
        调 LLM 让它给出 1~10 分的重要性。
        你也可以用简单规则替代。
        """
        system = "你是一个评分助手。"
        user = f"""请根据这件事对一个人未来生活的潜在影响，给出一个 1 到 10 的重要性评分，只返回数字。
事件：{text}
"""
        try:
            resp = self.llm_client.chat(system, user)
            # 提取数字
            digits = "".join(ch for ch in resp if ch.isdigit() or ch == ".")
            score = float(digits)
            score = max(1.0, min(score, 10.0))
        except Exception:
            score = 5.0
        return score

    def _recency_score(self, when: datetime) -> float:
        now = datetime.utcnow()
        dt = now - when
        halves = dt / self.recency_half_life
        return 0.5 ** halves

    # ---- 对外接口 ----

    def add(self, text: str) -> MemoryItem:
        importance = self._score_importance(text)
        emb = self._embed(text)
        item = MemoryItem(
            text=text,
            created_at=datetime.utcnow(),
            importance=importance,
            embedding=emb,
        )
        self.memories.append(item)
        self._importance_accumulator += importance
        return item

    def retrieve(self, query: str, k: int = 10) -> List[MemoryItem]:
        if not self.memories:
            return []

        query_emb = self._embed(query)
        embs = np.stack([m.embedding for m in self.memories])
        sims = cosine_similarity([query_emb], embs)[0]

        scores = []
        for m, sim in zip(self.memories, sims):
            recency = self._recency_score(m.created_at)
            score = 0.5 * sim + 0.3 * recency + 0.2 * (m.importance / 10.0)
            scores.append(score)

        ranked = sorted(zip(self.memories, scores), key=lambda x: x[1], reverse=True)
        return [m for m, _ in ranked[:k]]

    def maybe_reflect(self) -> List[MemoryItem]:
        """
        触发一次反思，把若干 recent memory 总结成高阶“反思记忆”再写回。
        """
        if self._importance_accumulator < self.reflection_threshold:
            return []

        self._importance_accumulator = 0.0
        recent = sorted(self.memories, key=lambda m: m.created_at, reverse=True)[:20]
        texts = "\n".join(f"- {m.text}" for m in recent)

        system = "你是一个善于自我反思的人。"
        user = f"""下面是你最近的一些经历，请总结 1~3 条你从中形成的长期看法或习惯性想法，用第一人称写，每条一行。

经历：
{texts}

请直接输出若干行，不要加序号。
"""

        resp = self.llm_client.chat(system, user)
        lines = [l.strip() for l in resp.split("\n") if l.strip()]
        new_items = []
        for line in lines:
            new_items.append(self.add(f"(反思) {line}"))
        return new_items
