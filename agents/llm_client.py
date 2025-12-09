# agents/llm_client.py

import os
import time
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
import requests

from config.settings import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)

PROXY_BASE_URL = "https://api.openai-proxy.org/v1"


class LLMClient:
    """
    简化版 Chat Completions 客户端（兼容 OpenAI 风格接口）。
    - 默认走代理基址 https://api.openai-proxy.org/v1
    - 默认使用用户提供的 API Key
    - 网络失败重试 3 次，失败后返回兜底文本
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        use_env_proxy: bool = False,
    ):
        load_dotenv()

        # 优先入参，其次环境变量，最后使用提供的默认值
        self.api_key = api_key or os.getenv(
            "CLOSEAI_API_KEY",
            DEFAULT_LLM_API_KEY,
        )
        if not self.api_key:
            raise ValueError("请在 .env 中设置 CLOSEAI_API_KEY")

        self.base_url = base_url or os.getenv("CLOSEAI_BASE_URL", DEFAULT_LLM_BASE_URL)

        self.model_name = model_name or os.getenv(
            "CLOSEAI_MODEL",
            DEFAULT_LLM_MODEL,
        )

        self.chat_url = f"{self.base_url}/chat/completions"
        self._proxies = None
        self._session = requests.Session()
        self._session.trust_env = use_env_proxy

    def _request_chat(
        self,
        messages: List[Dict[str, str]],
        enable_thinking: bool = False,
        temperature: float = 0.7,
        **extra_params: Any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if enable_thinking:
            payload["enable_thinking"] = True
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
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    raise RuntimeError(f"API error: {data['error']}")
                content = data["choices"][0]["message"]["content"]
                return content.strip()
            except requests.exceptions.RequestException as e:
                last_error = e
                print(f"[LLMClient] 请求失败，第 {attempt + 1} 次重试：{repr(e)}")
                time.sleep(1.0)
            except Exception as e:
                last_error = e
                print(f"[LLMClient] 解析响应失败：{repr(e)}")
                break

        print(f"[LLMClient] 调用失败，使用兜底结果。最后错误：{repr(last_error)}")
        return "【系统提示】本轮大模型调用失败，保持沉默或使用默认行为。"

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._request_chat(messages=messages, enable_thinking=False, temperature=temperature)

    def chat_thinking(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._request_chat(messages=messages, enable_thinking=True, temperature=temperature)
