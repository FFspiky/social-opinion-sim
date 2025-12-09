from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import math
import time
import random

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
    维护话题热度，混合基础统计与简化的 Hawkes 记忆项。
    """

    def __init__(self, topics: Sequence[str], hawkes_params: Optional[Dict[str, float]] = None):
        params = hawkes_params or {}
        self.alpha_v = params.get("alpha_v", params.get("alpha", 1.0))
        self.beta_c = params.get("beta_c", params.get("beta", 0.5))
        self.gamma_r = params.get("gamma_r", params.get("gamma", 0.3))
        # 训练集热度量级（可调），通过 heat_scale 做线性放大，便于对齐真实数据
        self.heat_scale = params.get("heat_scale", 1e4)
        # 双衰减核参数（与训练保持一致）
        self.H_base = params.get("H_base", 0.0)
        self.mu_fast = params.get("mu_fast", params.get("mu", 0.5))
        self.mu_slow = params.get("mu_slow", 0.2)
        self.lambda_fast = params.get("lambda_fast", params.get("lambda", 1.0))
        self.lambda_slow = params.get("lambda_slow", 0.3)
        # 保证衰减正且 fast > slow
        self.lambda_fast = max(self.lambda_fast, 1e-6)
        self.lambda_slow = max(self.lambda_slow, 1e-6)
        if self.lambda_slow >= self.lambda_fast:
            self.lambda_slow = max(self.lambda_fast * 0.8, 1e-6)

        self.topics: Dict[str, Dict[str, Any]] = {
            topic: {
                "heat": 0.0,
                "posts": [],
                "events": [],
                "per_step": {},  # t -> {"V": count, "C": 评论数, "R": reach}
                "mem_fast": 0.0,
                "mem_slow": 0.0,
                "last_time": None,
            } for topic in topics
        }

    def _compute_base(self, topic: str, current_time: int) -> float:
        tdata = self.topics[topic]
        stats = tdata["per_step"].get(current_time, {"V": 0, "C": 0, "R": 0})
        V = stats["V"]
        C = stats["C"]
        R = stats["R"]
        return (
            self.alpha_v * math.log(V + 1)
            + self.beta_c * math.log(C + 1)
            + self.gamma_r * math.log(R + 1)
        )

    def _compute_heat(self, topic: str, current_time: int) -> float:
        tdata = self.topics[topic]
        base = self._compute_base(topic, current_time)
        hawkes = self.H_base + self.mu_fast * tdata["mem_fast"] + self.mu_slow * tdata["mem_slow"]
        return (base + hawkes) * self.heat_scale

    def add_post(self, topic: str, post_content: str, current_time: int = 0, reach: float = 0.0) -> None:
        """
        记录新增帖子并刷新热度；reach 表示影响范围，可简单累加。
        """
        if topic not in self.topics:
            return
        tdata = self.topics[topic]
        tdata["posts"].append(post_content)
        tdata["events"].append(current_time)
        step_stats = tdata["per_step"].setdefault(current_time, {"V": 0, "C": 0, "R": 0})
        step_stats["V"] += 1
        step_stats["R"] += reach
        # 双衰减记忆更新
        last_time = tdata["last_time"]
        dt = (current_time - last_time) if last_time is not None else 0
        decay_fast = math.exp(-self.lambda_fast * dt) if dt > 0 else 1.0
        decay_slow = math.exp(-self.lambda_slow * dt) if dt > 0 else 1.0
        tdata["mem_fast"] = tdata["mem_fast"] * decay_fast + 1.0
        tdata["mem_slow"] = tdata["mem_slow"] * decay_slow + 1.0
        tdata["last_time"] = current_time

        tdata["heat"] = self._compute_heat(topic, current_time)

    def get_heat(self, topic: str) -> float:
        """获取当前热度；未知话题返回 0。"""
        return self.topics.get(topic, {}).get("heat", 0.0)


class SocialEnv:
    """
    社交环境：
    - G：关注关系图
    - agents：name -> Agent
    - posts：历史帖子列表
    - topic_manager：记录各话题热度
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
        self._agent_last_action: Dict[str, int] = {name: 0 for name in agents}

    def reset(self):
        self.posts = []
        self.t = 0
        self._next_post_id = 1
        if self._topics:
            self.topic_manager = TopicManager(self._topics, self._hawkes_params)
        self._agent_last_action = {name: 0 for name in self.agents}

    def _compute_reach(self, author: str) -> int:
        """简单地以关注入度作为传播影响力近似。"""
        try:
            return int(self.G.in_degree(author))
        except Exception:
            return 0

    def _add_post(
        self,
        author: str,
        text: str,
        sentiment: str,
        tag: str,
        target_post_id: Optional[int] = None,
        topic: Optional[str] = None,
    ):
        post = Post(
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
        self.posts.append(post)

        if self.topic_manager and topic:
            reach = self._compute_reach(author)
            self.topic_manager.add_post(topic, text, current_time=self.t, reach=reach)

    def step(self, pr_strategy=None, request_delay: float = 0.0):
        """推进一个时间步，按权重驱动 Agent 发声。"""
        self.t += 1
        new_posts: List[Post] = []
        observed = [
            {
                "id": p.id,
                "author": p.author,
                "text": p.text,
                "summary": p.text,
                "sentiment": p.sentiment,
                "tag": p.tag,
                "topic": p.topic,
            }
            for p in self.posts
            if p.time_step == self.t - 1
        ]

        # ---- 基于 Hawkes 参数 + 角色画像的权重调度 ----
        mu_fast = self._hawkes_params.get("mu_fast", 0.5) if self._hawkes_params else 0.5
        mu_slow = self._hawkes_params.get("mu_slow", 0.2) if self._hawkes_params else 0.2
        lambda_fast = self._hawkes_params.get("lambda_fast", 1.0) if self._hawkes_params else 1.0
        lambda_slow = self._hawkes_params.get("lambda_slow", 0.3) if self._hawkes_params else 0.3
        base_map = {
            "official_media": 0.35 * mu_slow,
            "kol": 0.30 * mu_fast,
            "troll": 0.15 * mu_fast,
            "defender": 0.10 * mu_fast,
            "crowd": 0.35 * mu_slow,
        }
        fast_roles = {"kol", "troll", "defender"}
        alpha_heat = 0.12  # 热度对出场概率的放大系数（更温和）

        weights = []
        names = []
        for name, agent in self.agents.items():
            role = getattr(agent, "role", "")
            base = base_map.get(role, 0.2 * mu_slow)
            lam = lambda_fast if role in fast_roles else lambda_slow
            last_t = self._agent_last_action.get(name, 0)
            dt = max(self.t - last_t, 0)
            decay = math.exp(-lam * dt) if dt > 0 else 1.0

            # 以 agent 关注的话题平均热度作为加权因子
            heat_boost = 1.0
            if self.topic_manager and getattr(agent, "topics", None):
                heats = [self.topic_manager.get_heat(tp) for tp in agent.topics]
                if heats:
                    avg_heat = sum(heats) / len(heats)
                    heat_boost = 1.0 + alpha_heat * math.log1p(avg_heat)

            w_eff = base * decay * heat_boost
            weights.append(max(w_eff, 0.0))
            names.append(name)

        active_agents: List[str] = []
        if weights and sum(weights) > 0:
            total_w = sum(weights)
            norm_w = [w / total_w for w in weights]
            target_k = max(1, int(len(names) * 0.4))  # 每步约 40% 参与，降低爆发
            pool = list(zip(names, norm_w))
            while pool and len(active_agents) < target_k:
                cand = random.choices(pool, weights=[w for _, w in pool], k=1)[0][0]
                active_agents.append(cand)
                pool = [(n, w) for n, w in pool if n != cand]
        else:
            active_agents = list(self.agents.keys())

        for name in active_agents:
            agent = self.agents[name]
            if request_delay > 0:
                time.sleep(request_delay)

            if pr_strategy and name == "BrandOfficial":
                action = pr_strategy.decide_brand_action(self.t, agent, observed)
            else:
                action = agent.decide_social_action(self.t, observed, environment=self)

            if action is None:
                continue

            act_type = action.get("action") or action.get("type")
            if act_type == "post":
                self._add_post(
                    author=name,
                    text=action.get("post_text", action.get("text", "")),
                    sentiment=action.get("sentiment", "NEUTRAL"),
                    tag=action.get("tag", "user"),
                    topic=action.get("topic"),
                )
                self._agent_last_action[name] = self.t
                new_posts.append(self.posts[-1])
            elif act_type == "retweet":
                target_id = action.get("target_post_id")
                self._add_post(
                    author=name,
                    text=action.get("post_text", action.get("text", "")),
                    sentiment=action.get("sentiment", "NEUTRAL"),
                    tag="retweet",
                    target_post_id=target_id,
                    topic=action.get("topic"),
                )
                self._agent_last_action[name] = self.t
                new_posts.append(self.posts[-1])
            else:
                continue
        return new_posts
