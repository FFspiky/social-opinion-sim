# agents/agent.py
from __future__ import annotations

from typing import Dict, Any, List, Optional

import random
import numpy as np

from config.styles import STYLE_GUIDE
from .memory import MemoryStream


class Agent:
    """
    多智能体舆论模拟中的单个 Agent。
    特点：
    - 角色类型支持 BrandOfficial / KOL / Troll / Defender / Crowd（上层传入的 role 决定 driver_mode）
    - 拥有人物设定 profile（背景、性格）与心理动力学参数
    - 拥有 MemoryStream（记忆系统）
    - 能根据看到的帖子在社交媒体上决定：post / retweet / silent，并支持 reflex/brain 双驱动
    """

    def __init__(
        self,
        name: str,
        role: str,
        profile: str,
        llm_client,
        topics: Optional[List[str]] = None,
        attention_weights: Optional[Dict[str, float]] = None,
    ):
        """
        :param name: Agent 名字（如 "BrandOfficial", "AngryUser1"）
        :param role: 角色类型（如 "BrandOfficial"/"KOL"/"Troll"/"Defender"/"Crowd"，用于确定 driver_mode）
        :param profile: 角色设定描述（自然语言）
        :param llm_client: 封装好的 LLMClient，拥有 chat / chat_thinking 方法
        :param topics: 关注的话题列表
        :param attention_weights: 话题注意力权重
        """
        self.name = name
        self.role = role
        self.profile = profile
        self.llm = llm_client
        self.topics = topics or []
        self.attention_weights = self._init_attention_weights(attention_weights)
        self.history: List[str] = []
        self.memory = MemoryStream(llm_client=llm_client)
        self.driver_mode = "brain" if self.role in ("BrandOfficial", "KOL", "Troll") else "reflex"
        self.psychology = {
            "conformity": np.random.beta(2, 5),
            "susceptibility": np.random.beta(2, 2),
            "ideology": np.random.random(),
            "risk_tolerance": 0.1 if self.role == "BrandOfficial" else np.random.beta(2, 5),
            "current_anger": 0.0,
        }
        self.trust_matrix = {
            "official": 0.9 if self.role == "BrandOfficial" else 0.6,
            "kol": 0.7,
            "rumor": 0.2,
        }

    def _init_attention_weights(
        self, provided_weights: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        初始化并规范化注意力权重，允许传入或自动生成随机权重。
        """
        weights = provided_weights or {topic: random.random() for topic in self.topics}
        if not weights and self.topics:
            weights = {topic: 1.0 for topic in self.topics}

        weights = {t: w for t, w in weights.items() if t in self.topics}
        for topic in self.topics:
            weights.setdefault(topic, random.random())

        return self._normalize_attention(weights)

    def _normalize_attention(self, weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(w, 0.0) for w in weights.values())
        if total <= 0:
            if not self.topics:
                return {}
            uniform = 1.0 / len(self.topics)
            return {topic: uniform for topic in self.topics}
        return {topic: max(w, 0.0) / total for topic, w in weights.items()}

    def update_attention(self, new_attention_weights: Dict[str, float]):
        """
        更新代理对各话题的注意力权重。
        """
        for topic in new_attention_weights:
            if topic not in self.topics:
                self.topics.append(topic)
        merged = {**self.attention_weights, **new_attention_weights}
        self.attention_weights = self._normalize_attention(merged)

    def select_topic(self, environment=None, temperature: float = 1.2) -> Optional[str]:
        """
        按有限理性资源分配选择话题：兴趣(interest) + 社会关系(soc) + 热度(heat) + 噪声，Softmax(温度)。
        """
        if not self.topics:
            return None

        # 兴趣：使用注意力权重近似兴趣向量
        interest = np.array([self.attention_weights.get(t, 0.0) for t in self.topics], dtype=float)
        if interest.sum() <= 0:
            interest = np.ones(len(self.topics)) / len(self.topics)
        else:
            interest = interest / interest.sum()

        # 社会关系：关注者最近一跳的帖子中同话题的占比
        soc = np.zeros(len(self.topics), dtype=float)
        if environment is not None and hasattr(environment, "G") and hasattr(environment, "posts"):
            for idx, topic in enumerate(self.topics):
                # 统计上一时间步里，关注对象在该话题的发帖数
                follows = list(environment.G.successors(self.name)) if environment.G.has_node(self.name) else []
                recent = [
                    p for p in environment.posts
                    if p.author in follows and p.time_step == environment.t - 1 and p.topic == topic
                ]
                total_recent = [
                    p for p in environment.posts
                    if p.author in follows and p.time_step == environment.t - 1
                ]
                soc[idx] = (len(recent) / max(len(total_recent), 1)) if total_recent else 0.0
        if soc.sum() > 0:
            soc = soc / soc.sum()

        # 热度：来自 topic_manager
        heat = np.zeros(len(self.topics), dtype=float)
        if environment is not None and getattr(environment, "topic_manager", None):
            tm = environment.topic_manager
            heat = np.array([tm.get_heat(t) for t in self.topics], dtype=float)
            if heat.sum() > 0:
                heat = heat / heat.sum()

        # 随机噪声，防止概率塌缩
        noise = np.random.random(len(self.topics))
        noise = noise / noise.sum()

        # 效用函数 U = w1*interest + w2*soc + w3*log(heat+1) + w4*noise
        log_heat = np.log(heat + 1e-6)  # 避免 log(0)
        utility = 0.5 * interest + 0.2 * soc + 0.2 * log_heat + 0.1 * noise

        # Softmax with temperature
        scores = utility / max(temperature, 1e-6)
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()
        return np.random.choice(self.topics, p=probs)

    def interact(self, environment):
        """
        代理根据当前关注的话题与环境互动，并记录历史。
        """
        topic = self.select_topic()
        if topic is None:
            return None
        print(f"Agent {self.name} interacts with topic {topic}")
        self.history.append(topic)
        # 优先通过环境记录话题互动，若无该接口则尝试直接写入话题管理器
        if hasattr(environment, "record_topic_interaction"):
            environment.record_topic_interaction(topic, f"{self.name} interacted with {topic}")
        elif hasattr(environment, "add_post"):
            environment.add_post(topic, f"{self.name} interacted with {topic}")
        return topic

    # ------------------------------------------------------------------
    # 记忆相关：观察外界事件
    # ------------------------------------------------------------------

    def observe(self, observation: str):
        """
        将一段文本写入记忆流；用于记录“自己发了什么/看到的关键事件”。
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
        forced_topic: Optional[str] = None,
    ) -> str:
        """
        构造让 Agent 在社交媒体上决策的 prompt。
        observed_posts 形如：
        [
          {"id": 1, "author": "Media1", "text": "...", "summary": "...", "sentiment": "NEGATIVE", "tag": "rumor"},
          ...
        ]
        """
        query = " ".join(self.topics) if self.topics else "近期热点 话题 热度 舆情"
        mem_items = self.memory.retrieve(query=query, k=5)
        mem_text = "\n".join(f"- {m.text}" for m in mem_items) or "（暂无重要的相关记忆）"

        topics_str = ", ".join(self.topics) if self.topics else "未给出"
        topic_hint = forced_topic or (self.topics[0] if self.topics else "未知话题")

        posts_str = "\n".join(
            f"- 来自 {p['author']}：{p['summary']}（情绪：{p['sentiment']}，标签：{p['tag']}）"
            for p in observed_posts
        ) or "（此时间步你没有看到任何新帖子）"

        prompt = f"""
你的名字：{self.name}
你的角色类型：{self.role}
你的背景和性格设定：{self.profile}
你当前关注的话题：{topics_str}

你目前记得的与热点话题相关的信息：
{mem_text}

现在是时间步 {t}，你在社交媒体上看到了这些新帖子（来自你关注的人）：
{posts_str}

你需要根据自己的角色和立场，决定是否发声，并选择要聚焦的话题。请注意：
1. 只使用简体中文表达。
2. 行为类型只能是：发原创帖（post）、转发已有帖子（retweet）、或保持沉默（silent）。
3. 如发帖/转发，请用符合角色设定的语气，写 1~2 句内容，并注明你聚焦的话题（topic）。优先围绕话题：{topic_hint}
输出格式要求（必须是合法 JSON）：
{{
  "action": "post" 或 "retweet" 或 "silent",
  "post_text": "若 action 不是 silent，这里写 1~2 句中文；若 silent，可为空字符串",
  "sentiment": "POSITIVE" 或 "NEGATIVE" 或 "NEUTRAL",
  "target_post_id": "若是 retweet，写要转发的帖子的 id（数字）；否则为 null",
  "topic": "选填，聚焦的话题"
}}
"""
        return prompt

    # ------------------------------------------------------------------
    # 对外接口：在社交环境中的一次决策
    # ------------------------------------------------------------------

    def decide_social_action(
        self,
        t: int,
        observed_posts: List[Dict[str, Any]],
        environment=None,
    ) -> Dict[str, Any]:
        """
        根据当前时间步 t 和能看到的帖子列表，返回一个动作：
        {
          "action": "post" / "retweet" / "silent",
          "post_text": "...",
          "sentiment": "POSITIVE" / "NEGATIVE" / "NEUTRAL",
          "target_post_id": int 或 None,
          "topic": 可选话题
        }
        """
        role_prompts = {
            "BrandOfficial": (
                "你是权威媒体/官方账号，语气正式客观，强调核实、等待通报，不信谣不传谣。"
            ),
            "KOL": (
                "你是微博意见领袖/营销号，带节奏、善用反问和悬念，语气接地气，情绪化以求流量。"
            ),
            "Troll": (
                "你是极端情绪用户/杠精，发言尖锐、嘲讽、攻击或阴阳怪气，擅长挑衅。"
            ),
            "Defender": (
                "你是死忠粉/护卫队，极度护短，控评、呼吁理性，反击黑子，充满爱意或防御性。"
            ),
            "Crowd": (
                "你是吃瓜群众，超短评、跟风、好奇打卡，立场摇摆，偶尔只@好友或发表情。"
            ),
        }
        role_hint = role_prompts.get(self.role, "你是一个普通用户，保持简洁、多样化表达，避免复读。")
        style = STYLE_GUIDE.get(self.role, STYLE_GUIDE.get("Crowd", {}))
        role_description = style.get("description", "普通用户")
        keywords = ", ".join(style.get("keywords", []))
        sentence_patterns = " | ".join(style.get("sentence_patterns", []))
        tone_instruction = style.get("tone_instruction", "")
        style_block = (
            "【语言风格要求】\n"
            f"你现在的身份是微博上的【{role_description}】。请严格遵守以下说话方式：\n"
            f"1. 必须包含关键词（选2-3个）: {keywords}\n"
            f"2. 推荐句式参考: {sentence_patterns}\n"
            f"3. 语气指令: {tone_instruction}"
        )
        system = (
            f"{role_hint}"
            " 你需要根据记忆和看到的帖子，决定是否发声。"
            " 严格输出 JSON，不要输出多余文字。"
            " 所有内容必须使用简体中文。"
        )
        system = f"{system}\n{style_block}"
        forced_topic = self.select_topic(environment=environment)
        user = self._build_social_prompt(t, observed_posts, forced_topic=forced_topic)

        resp = self.llm.chat(system, user)

        # 尝试解析 JSON，失败则回退为沉默
        import json

        try:
            action = json.loads(resp)
            if action.get("action") not in ("post", "retweet", "silent"):
                raise ValueError("invalid action")
            if action.get("action") == "silent":
                action.setdefault("post_text", "")
                action.setdefault("sentiment", "NEUTRAL")
                action["target_post_id"] = None
            else:
                action.setdefault("post_text", "")
                action.setdefault("sentiment", "NEUTRAL")
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
                "topic": forced_topic,
            }

        # 如 LLM 未提供话题，则回填选定话题
        if not action.get("topic"):
            action["topic"] = forced_topic

        return action

    def _calculate_reflex_action(self, post, env_context):
        """
        反射型智能体基于概率的快速决策：返回 ("REPOST"/"SILENCE", mode)。
        """
        context = env_context or {}
        global_tension = context.get("global_tension", 0.5)

        emotional_boost = self.psychology.get("susceptibility", 0.0) * global_tension
        heat = getattr(post, "heat", 0.0) or 0.0
        social_boost = self.psychology.get("conformity", 0.0) * np.log1p(heat)
        trust_factor = self.trust_matrix.get(getattr(post, "author_role", None), 0.5)

        risk_penalty = 0.0
        verified = getattr(post, "is_verified", False)
        low_risk_tolerance = self.psychology.get("risk_tolerance", 0.0) < 0.3
        too_angry = self.psychology.get("current_anger", 0.0) > 0.7
        if not verified and low_risk_tolerance and not too_angry:
            risk_penalty = 5.0

        logit = (0.5 + emotional_boost + social_boost) * trust_factor - risk_penalty
        prob_repost = 1 / (1 + np.exp(-(logit - 2.0)))

        dice = np.random.random()
        if dice < prob_repost:
            return ("REPOST", "forward_only")
        return ("SILENCE", None)

    def _call_llm_decision(self, post, env_context=None):
        """
        深度（brain）模式的决策，封装 LLM 调用；默认无法生成时保持沉默。
        """
        if not hasattr(self.llm, "chat"):
            return ("SILENCE", None)

        context = env_context or {}
        global_tension = context.get("global_tension", 0.5)
        author_role = getattr(post, "author_role", "unknown")
        content = getattr(post, "text", "") or getattr(post, "content", "")
        verified = getattr(post, "is_verified", False)
        heat = getattr(post, "heat", 0.0)

        system = (
            f"你是 {self.name}（角色：{self.role}），需要深思熟虑地决定是否转发或沉默。"
            " 只输出 REPOST 或 SILENCE。"
        )
        user = (
            f"帖子作者角色: {author_role}\n"
            f"热度: {heat}\n"
            f"是否已证实: {verified}\n"
            f"全局紧张度: {global_tension}\n"
            f"帖子内容: {content}"
        )
        try:
            resp = self.llm.chat(system, user)
            normalized = str(resp).strip().upper()
            if normalized.startswith("REPOST"):
                return ("REPOST", "llm_brain")
        except Exception:
            pass
        return ("SILENCE", None)

    def decide_action(self, post, env_context=None):
        """
        混合驱动决策入口：reflex 模式走概率，brain 模式走 LLM。
        """
        if self.driver_mode == "reflex":
            return self._calculate_reflex_action(post, env_context)
        return self._call_llm_decision(post, env_context)
