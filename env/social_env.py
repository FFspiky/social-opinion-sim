from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import time
import math

import networkx as nx


@dataclass
class Post:
    id: int
    author: str
    text: str
    sentiment: str  # "POSITIVE", "NEGATIVE", "NEUTRAL"
    tag: str        # "rumor", "official", "user", ...
    time_step: int
    target_post_id: Optional[int] = None
    topic: Optional[str] = None


class TopicManager:
    """
    话题管理器：维护帖子、热度，并采用基于霍克斯思想的自激热度模型。
    """

    def __init__(self, topics: Sequence[str], hawkes_params: Optional[Dict[str, float]] = None):
        params = hawkes_params or {}
        self.alpha_v = params.get("alpha_v", params.get("alpha", 1.0))
        self.beta_c = params.get("beta_c", params.get("beta", 0.5))
        self.gamma_r = params.get("gamma_r", params.get("gamma", 0.3))
        self.mu = params.get("mu", 0.1)          # 激发强度
        self.decay = params.get("decay", params.get("lambda", 0.5))  # 衰减因子

        self.topics: Dict[str, Dict[str, Any]] = {
            topic: {
                "heat": 0.0,
                "posts": [],
                "events": [],
                "per_step": {},  # t -> {"V": count, "C": 评论数, "R": reach}
            } for topic in topics
        }

    def _compute_heat(self, topic: str, current_time: int) -> float:
        tdata = self.topics[topic]
        stats = tdata["per_step"].get(current_time, {"V": 0, "C": 0, "R": 0})
        V = stats["V"]
        C = stats["C"]
        R = stats["R"]

        base = (
            self.alpha_v * math.log(V + 1)
            + self.beta_c * math.log(C + 1)
            + self.gamma_r * math.log(R + 1)
        )
        hawkes = 0.0
        for ti in tdata["events"]:
            if ti < current_time:
                hawkes += self.mu * math.exp(-self.decay * (current_time - ti))
        return base + hawkes

    def add_post(self, topic: str, post_content: str, current_time: int = 0, reach: float = 0.0) -> None:
        """
        记录帖子并更新热度。reach 表示作者影响力（如关注者数）。
        """
        if topic not in self.topics:
            return
        tdata = self.topics[topic]
        tdata["posts"].append(post_content)
        tdata["events"].append(current_time)
        step_stats = tdata["per_step"].setdefault(current_time, {"V": 0, "C": 0, "R": 0})
        step_stats["V"] += 1
        step_stats["R"] += reach

        tdata["heat"] = self._compute_heat(topic, current_time)

    def get_heat(self, topic: str) -> float:
        """
        获取话题当前热度，未知话题返回 0。
        """
        return self.topics.get(topic, {}).get("heat", 0.0)


class SocialEnv:
    """
    社交媒体模拟环境：
    - G：有向关注图
    - agents：name -> Agent
    - posts：历史帖子列表
    - t：当前时间步
    - topic_manager：管理话题热度与帖子
    """

    def __init__(
        self,
        agents: Dict[str, Any],
        graph: nx.DiGraph,
        topics: Optional[Sequence[str]] = None,
        hawkes_params: Optional[Dict[str, float]] = None,
    ):
        self.agents = agents
        self.G = graph
        self.posts: List[Post] = []
        self.t = 0
        self._next_post_id = 1
        self._topics = list(topics) if topics else []
        self._hawkes_params = hawkes_params
        self.topic_manager: Optional[TopicManager] = (
            TopicManager(self._topics, hawkes_params) if self._topics else None
        )

    def reset(self):
        self.posts = []
        self.t = 0
        self._next_post_id = 1
        if self._topics:
            self.topic_manager = TopicManager(self._topics, self._hawkes_params)

    def _compute_reach(self, author: str) -> int:
        """简单用粉丝数（入度）作为 reach 近似。"""
        if not self.G or author not in self.G:
            return 0
        return self.G.in_degree(author)

    # ---- 内部工具 ----

    def _add_post(
        self,
        author: str,
        text: str,
        sentiment: str,
        tag: str,
        target_post_id: Optional[int] = None,
        topic: Optional[str] = None,
    ) -> Post:
        p = Post(
            id=self._next_post_id,
            author=author,
            text=text,
            sentiment=sentiment,
            tag=tag,
            time_step=self.t,
            target_post_id=target_post_id,
            topic=topic,
        )
        self._next_post_id += 1
        self.posts.append(p)

        if self.topic_manager and topic:
            reach = self._compute_reach(author)
            self.topic_manager.add_post(topic, text, current_time=self.t, reach=reach)

        return p

    def record_topic_interaction(self, topic: str, content: str = "") -> None:
        """
        允许外部调用（如 Agent）直接为话题记录一次互动，提升热度。
        """
        if self.topic_manager:
            self.topic_manager.add_post(topic, content or f"interaction on {topic}", current_time=self.t, reach=0)

    def get_topic_heat(self, topic: str) -> float:
        """
        获取指定话题热度。
        """
        if not self.topic_manager:
            return 0
        return self.topic_manager.get_heat(topic)

    def get_visible_posts_for(self, agent_name: str) -> List[Dict[str, Any]]:
        """
        在时间步 t，agent_name 可以看到 t-1 时刻其关注对象发布的帖子。
        """
        following = list(self.G.successors(agent_name))
        recent_posts = [p for p in self.posts if p.author in following and p.time_step == self.t - 1]
        result = []
        for p in recent_posts:
            result.append({
                "id": p.id,
                "author": p.author,
                "text": p.text,
                "summary": p.text[:50],
                "sentiment": p.sentiment,
                "tag": p.tag,
                "topic": p.topic,
            })
        return result

    # ---- 推进一个时间步 ----

    def step(self, request_delay: float = 0.0):
        """
        执行一个时间步：所有 Agent 基于可见帖子发言，用于话题热度与传播模拟。
        """
        self.t += 1
        new_posts: List[Post] = []

        for name, agent in self.agents.items():
            observed = self.get_visible_posts_for(name)
            if not observed:
                continue

            action = agent.decide_social_action(self.t, observed, environment=self)
            if action["action"] == "silent":
                continue

            p = self._add_post(
                author=name,
                text=action["post_text"],
                sentiment=action["sentiment"],
                tag="user",
                target_post_id=action.get("target_post_id"),
                topic=action.get("topic"),
            )
            new_posts.append(p)
            agent.observe(f"我在时间 {self.t} 在社交媒体上发了：{p.text}")
            if request_delay > 0:
                time.sleep(request_delay)

        return new_posts
