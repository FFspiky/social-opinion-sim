# pr_strategies/strategies.py
from __future__ import annotations
from typing import Dict, Any, List


class BasePRStrategy:
    """
    所有公关策略的基类。
    """

    def __init__(self, brand_name: str):
        self.brand_name = brand_name

    def decide_brand_action(
        self,
        t: int,
        agent,
        observed_posts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        raise NotImplementedError


class DoNothingStrategy(BasePRStrategy):
    """
    S0：完全不作为。
    """

    def decide_brand_action(self, t, agent, observed_posts):
        return {
            "action": "silent",
            "post_text": "",
            "sentiment": "NEUTRAL",
            "target_post_id": None,
        }


class DelayedApologyStrategy(BasePRStrategy):
    """
    S1：在时间步 2 发一次道歉声明。
    """

    def __init__(self, brand_name: str):
        super().__init__(brand_name)
        self.has_apologized = False

    def decide_brand_action(self, t, agent, observed_posts):
        if t == 2 and not self.has_apologized:
            self.has_apologized = True
            system = "你是一个品牌官方公关，语气真诚，不推卸责任。"
            user = """写一条简短的公开道歉声明，承认数据泄露问题，
向用户道歉，并承诺尽快查明原因和做好补救，不要太长。"""
            text = agent.llm.chat(system, user)
            return {
                "action": "post",
                "post_text": text,
                "sentiment": "NEUTRAL",
                "target_post_id": None,
            }
        return {
            "action": "silent",
            "post_text": "",
            "sentiment": "NEUTRAL",
            "target_post_id": None,
        }


class FastClarifyStrategy(BasePRStrategy):
    """
    S2：一开始就声明 + 间隔更新。
    """

    def __init__(self, brand_name: str, update_interval: int = 5):
        super().__init__(brand_name)
        self.update_interval = update_interval
        self.has_initial = False

    def decide_brand_action(self, t, agent, observed_posts):
        if not self.has_initial and t == 1:
            self.has_initial = True
            system = "你是一个品牌官方公关，既要诚恳也要专业。"
            user = """写一条简短的初始声明：
- 承认关注到数据泄露报道
- 表示正在核查
- 承诺会第一时间公布进展
字数在 2~3 句话。"""
            text = agent.llm.chat(system, user)
            return {
                "action": "post",
                "post_text": text,
                "sentiment": "NEUTRAL",
                "target_post_id": None,
            }

        if self.has_initial and t % self.update_interval == 0:
            system = "你是品牌官方公关，用简短语言给用户更新调查进展。"
            user = "写 1~2 句话，说明调查进展和后续计划，避免重复完全一样的话。"
            text = agent.llm.chat(system, user)
            return {
                "action": "post",
                "post_text": text,
                "sentiment": "NEUTRAL",
                "target_post_id": None,
            }

        return {
            "action": "silent",
            "post_text": "",
            "sentiment": "NEUTRAL",
            "target_post_id": None,
        }
