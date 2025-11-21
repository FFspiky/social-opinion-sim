# env/social_env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

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


class SocialEnv:
    """
    社交媒体模拟环境：
    - G: 有向关注图
    - agents: name -> Agent
    - posts: 所有历史帖子
    - t: 当前时间步
    """

    def __init__(self, agents: Dict[str, Any], graph: nx.DiGraph):
        self.agents = agents
        self.G = graph
        self.posts: List[Post] = []
        self.t = 0
        self._next_post_id = 1

    def reset(self):
        self.posts = []
        self.t = 0
        self._next_post_id = 1

    # ---- 内部工具 ----

    def _add_post(
        self,
        author: str,
        text: str,
        sentiment: str,
        tag: str,
        target_post_id: Optional[int] = None,
    ) -> Post:
        p = Post(
            id=self._next_post_id,
            author=author,
            text=text,
            sentiment=sentiment,
            tag=tag,
            time_step=self.t,
            target_post_id=target_post_id,
        )
        self._next_post_id += 1
        self.posts.append(p)
        return p

    def get_visible_posts_for(self, agent_name: str) -> List[Dict[str, Any]]:
        """
        在时间步 t，agent_name 可以看到 t-1 时关注的人发的帖子。
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
            })
        return result

    # ---- 关键：推进一步 ----

    def step(self, pr_strategy=None):
        """
        执行一个时间步：
        1. 品牌官方先根据策略发言
        2. 其它 agent 再根据看到的内容发言
        """
        self.t += 1
        new_posts: List[Post] = []

        # 1）品牌官方
        if pr_strategy is not None and hasattr(pr_strategy, "brand_name"):
            brand_name = pr_strategy.brand_name
            brand_agent = self.agents[brand_name]
            observed = self.get_visible_posts_for(brand_name)
            brand_action = pr_strategy.decide_brand_action(
                t=self.t,
                agent=brand_agent,
                observed_posts=observed,
            )
            if brand_action and brand_action["action"] != "silent":
                p = self._add_post(
                    author=brand_name,
                    text=brand_action["post_text"],
                    sentiment=brand_action.get("sentiment", "NEUTRAL"),
                    tag="official",
                    target_post_id=brand_action.get("target_post_id"),
                )
                new_posts.append(p)
                brand_agent.observe(f"我在时间 {self.t} 发了官方声明：{p.text}")

        # 2）其他用户
        for name, agent in self.agents.items():
            if pr_strategy is not None and name == getattr(pr_strategy, "brand_name", None):
                continue

            observed = self.get_visible_posts_for(name)
            if not observed:
                continue

            action = agent.decide_social_action(self.t, observed)
            if action["action"] == "silent":
                continue

            p = self._add_post(
                author=name,
                text=action["post_text"],
                sentiment=action["sentiment"],
                tag="user",
                target_post_id=action.get("target_post_id"),
            )
            new_posts.append(p)
            agent.observe(f"我在时间 {self.t} 在社交媒体上发了：{p.text}")

        return new_posts
