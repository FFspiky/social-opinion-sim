# agents/llm_client.py

import os
import time
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
import requests


class LLMClient:
    """
    直接使用 SiliconFlow Chat Completions API 的客户端。
    支持 deepseek-ai/DeepSeek-V3.1-Terminus 的 enable_thinking 参数。

    关键改动：
    - 对网络错误 / 5xx 增加了最多 3 次重试
    - 如果仍然失败，不再抛异常，而是返回一个“安全兜底文本”
      * 对 Agent 来说：解析失败会自动回退为 silent 行为
      * 对总结报告来说：会看到一段“本轮 LLM 调用失败”的提示文本
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        use_env_proxy: bool = False,
    ):
        # 加载 .env
        load_dotenv()

        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("请在 .env 中设置 SILICONFLOW_API_KEY")

        self.base_url = base_url or os.getenv(
            "SILICONFLOW_BASE_URL",
            "https://api.siliconflow.cn/v1",
        )

        self.model_name = model_name or os.getenv(
            "SILICONFLOW_MODEL",
            "deepseek-ai/DeepSeek-V3.1-Terminus",
        )

        # Chat Completions 端点（兼容 OpenAI 风格）
        self.chat_url = f"{self.base_url}/chat/completions"
        # 控制是否使用系统代理。默认关闭以避免 ProxyError。
        self._proxies = None
        # 使用 requests.Session 以便控制 trust_env；默认不信任系统代理
        self._session = requests.Session()
        self._session.trust_env = use_env_proxy

    # ---------------- 核心请求封装 ---------------- #

    def _request_chat(
        self,
        messages: List[Dict[str, str]],
        enable_thinking: bool = False,
        temperature: float = 0.7,
        **extra_params: Any,
    ) -> str:
        """
        直接 POST 到 SiliconFlow Chat Completions。
        enable_thinking 控制思考模式开关：
        - False：非思考模式（默认）
        - True：思考模式

        增强：
        - 最多重试 3 次
        - 所有 requests 异常 / HTTPError 不再向外抛出，使用兜底文本
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "enable_thinking": enable_thinking,
        }

        # 允许额外参数往下透传
        payload.update(extra_params)

        last_error: Optional[Exception] = None

        for attempt in range(3):
            try:
                resp = self._session.post(
                    self.chat_url,
                    json=payload,
                    headers=headers,
                    timeout=10000,
                    proxies=self._proxies,
                )
                # 如果不是 2xx，会抛 HTTPError
                resp.raise_for_status()
                data = resp.json()

                # 简单防御：如果接口返回了 error 字段
                if "error" in data:
                    raise RuntimeError(f"SiliconFlow API error: {data['error']}")

                content = data["choices"][0]["message"]["content"]
                return content.strip()

            except requests.exceptions.RequestException as e:
                # 网络错误 / 5xx / 超时 都会到这里
                last_error = e
                print(
                    f"[LLMClient] 调用 SiliconFlow 失败，第 {attempt + 1} 次重试：{repr(e)}"
                )
                # 简单退避等待一会儿再试
                time.sleep(1.0)

            except Exception as e:
                # 其它解析类错误
                last_error = e
                print(f"[LLMClient] 解析 LLM 返回结果失败：{repr(e)}")
                break

        # 如果走到这里，说明三次尝试都失败了，用兜底文本
        print(f"[LLMClient] 最终放弃本次 LLM 调用，使用兜底结果。最后错误：{repr(last_error)}")

        # 兜底返回内容说明：
        # - 对 Agent.decide_social_action 来说：
        #     json.loads 会失败，最后 fall back 成为 silent 行为（不发言）
        # - 对 记忆重要度打分来说：
        #     _score_importance 只会从中提取数字，提取不到则用默认 5.0
        # - 对 总结报告 / 反思来说：
        #     页面上会看到这段中文提示，而不是直接崩溃
        return "【系统提示】本轮大模型调用失败，保持沉默或使用默认行为。"

    # ---------------- 对外接口：非思考模式 ---------------- #

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """
        默认：非思考模式（enable_thinking = False）
        用于绝大多数 Agent 的发帖/评分逻辑。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._request_chat(
            messages=messages,
            enable_thinking=False,
            temperature=temperature,
        )

    # ---------------- 对外接口：思考模式 ---------------- #

    def chat_thinking(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """
        思考模式：enable_thinking = True
        建议只在 Memory 反思 / PR 策略规划 / 实验总结报告等场景使用。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._request_chat(
            messages=messages,
            enable_thinking=True,
            temperature=temperature,
        )
