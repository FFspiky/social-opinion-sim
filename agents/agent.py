# agents/agent.py
from __future__ import annotations

from typing import Dict, Any, List

from .memory import MemoryStream


class Agent:
    """
    多智能体舆论模拟中的单个 Agent。

    特点：
    - 拥有角色类型（brand_official / angry_user / neutral_user / fan_user / media 等）
    - 拥有人物设定 profile（背景、性格）
    - 拥有 MemoryStream（记忆系统）
    - 能根据看到的帖子，在社交媒体上决定：post / retweet / silent
    """

    def __init__(
        self,
        name: str,
        role: str,
        profile: str,
        llm_client,
    ):
        """
        :param name: Agent 名字（如 "BrandOfficial", "AngryUser1"）
        :param role: 角色类型（如 "brand_official", "angry_user"）
        :param profile: 角色设定描述（自然语言）
        :param llm_client: 封装好的 LLMClient，拥有 chat / chat_thinking 方法
        """
        self.name = name
        self.role = role
        self.profile = profile
        self.llm = llm_client
        self.memory = MemoryStream(llm_client=llm_client)

    # ------------------------------------------------------------------
    # 记忆相关：观察外界事件
    # ------------------------------------------------------------------

    def observe(self, observation: str):
        """
        将一段文本写入记忆流。
        在当前简化版中，我们主要把“自己发了什么 / 看到的关键事件”写进去。
        """
        self.memory.add(observation)
        self.memory.maybe_reflect()

    # ------------------------------------------------------------------
    # 内部工具：构造给 LLM 的 prompt
    # ------------------------------------------------------------------

    def _build_social_prompt(
        self,
        t: int,
        observed_posts: List[Dict[str, Any]],
    ) -> str:
        """
        构造让 Agent 在社交媒体上决策用的 prompt。

        observed_posts 的结构大致为：
        [
          {
            "id": 1,
            "author": "Media1",
            "text": "...",
            "summary": "...",
            "sentiment": "NEGATIVE",
            "tag": "rumor"
          },
          ...
        ]
        """
        # 从记忆中检索与“舆情、公关、星光电子事件”相关的记忆
        mem_items = self.memory.retrieve(
            query="星光电子 数据泄露 危机公关 社交媒体 舆论 事件",
            k=5,
        )
        mem_text = "\n".join(f"- {m.text}" for m in mem_items) or "（暂无特别重要的相关记忆）"

        posts_str = "\n".join(
            f"- 来自 {p['author']}：{p['summary']}（情绪：{p['sentiment']}，标签：{p['tag']}）"
            for p in observed_posts
        ) or "（此时间步你没有看到任何新帖子）"

        prompt = f"""
你的名字是：{self.name}
你的角色类型是：{self.role}
你的背景和性格设定如下：
{self.profile}

你目前记得这些与“星光电子数据泄露事件”相关的事情：
{mem_text}

现在是时间步 {t}，你在社交媒体上看到了这些新帖子（来自你关注的人）：
{posts_str}

你需要根据自己的角色和立场，决定此时是否在平台上发声，以及如何发声。

请注意：
1. 你是生活在中文互联网环境中的用户，请只使用简体中文表达，不要出现任何英文单词或句子。
2. 你在社交媒体上的行为类型只能是：发原创帖（post）、转发某条已有帖子（retweet）、或者保持沉默（silent）。
3. 如果你选择发帖或转发，请用符合你角色设定的语气写 1~2 句内容。

输出格式要求（非常重要）：
- 只能输出一个 JSON 对象，不要添加任何解释性文字。
- JSON 的字段为：
  {{
    "action": "post" 或 "retweet" 或 "silent",
    "post_text": "如果 action 不是 silent，这里写你要发的中文内容，1~2 句；如果是 silent，则可以是空字符串",
    "sentiment": "POSITIVE" 或 "NEGATIVE" 或 "NEUTRAL",
    "target_post_id": "如果是 retweet，这里写你要转发的帖子的 id（数字）；否则为 null"
  }}

请严格按照上述 JSON 结构输出，并确保是合法的 JSON（双引号、逗号位置正确）。
"""
        return prompt

    # ------------------------------------------------------------------
    # 对外接口：在社交环境中的一次决策
    # ------------------------------------------------------------------

    def decide_social_action(
        self,
        t: int,
        observed_posts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        根据当前时间步 t 和能看到的帖子列表，返回一个动作：
        {
          "action": "post" / "retweet" / "silent",
          "post_text": "...",
          "sentiment": "POSITIVE" / "NEGATIVE" / "NEUTRAL",
          "target_post_id": int 或 None
        }
        """
        # system prompt：强约束中文 + JSON 输出
        system = (
            "你是一个在中文社交媒体上发言的普通用户或账号。"
            "你需要根据用户记忆和看到的帖子，决定是否发声。"
            "请严格按照用户提供的 JSON 格式输出结果，不要输出多余文字。"
            "所有内容必须使用简体中文，不要出现任何英文单词或句子。"
        )
        user = self._build_social_prompt(t, observed_posts)

        resp = self.llm.chat(system, user)

        # 尝试解析 JSON，失败则回退为沉默
        import json
        try:
            action = json.loads(resp)
            # 做一点简单的健壮性处理
            if action.get("action") not in ("post", "retweet", "silent"):
                raise ValueError("invalid action")
            if action.get("action") == "silent":
                action.setdefault("post_text", "")
                action.setdefault("sentiment", "NEUTRAL")
                action["target_post_id"] = None
            else:
                action.setdefault("post_text", "")
                action.setdefault("sentiment", "NEUTRAL")
                # target_post_id 如果不是数字，统一置为 None
                tgt = action.get("target_post_id")
                if isinstance(tgt, str) and tgt.isdigit():
                    action["target_post_id"] = int(tgt)
                elif isinstance(tgt, int):
                    action["target_post_id"] = tgt
                else:
                    action["target_post_id"] = None
        except Exception:
            action = {
                "action": "silent",
                "post_text": "",
                "sentiment": "NEUTRAL",
                "target_post_id": None,
            }

        return action
