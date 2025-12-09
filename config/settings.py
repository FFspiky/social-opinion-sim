from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _int_from_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# 路径与数据源
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "datasets_huoju_norm")
FALLBACK_DATA_DIR = "dataset_peak350"
DEFAULT_REAL_DATA_PATH = Path(
    os.getenv(
        "REAL_DATA_PATH",
        PROJECT_ROOT
        / ".."
        / "huoju"
        / "dataset_peak350"
        / "classified_events_35_2024Q1-Q4_peak350_v2.csv",
    )
).resolve()

# 默认话题与随机种子
DEFAULT_TOPICS = [
    "哈工大你玩真的啊",
    "为啥网上的药比实体药店更便宜",
    "晒晒家乡隐藏款土特产",
    "太空发快递可以当日达了",
    "春晚节目",
]
DEFAULT_SIM_SEED = _int_from_env("SIM_SEED", 42)

# Hawkes 模型默认参数
DEFAULT_HAWKES_PARAMS: Dict[str, float] = {
    "mu_fast": 0.5889573726987519,
    "mu_slow": 0.19392023078364723,
    "H_base": 1.8134757737318496,
    "lambda_fast": 4.996698275314672,
    "lambda_slow": 0.663701052879747,
}
HAWKES_PARAM_PATH = Path(
    os.getenv("HAWKES_PARAM_PATH", PROJECT_ROOT / "artifacts" / "hawkes_params.json")
)

# LLM 配置（保留默认明文，便于实验快速切换）
DEFAULT_LLM_API_KEY = os.getenv(
    "CLOSEAI_API_KEY",
    "sk-IzBkdFXS1V25hei9N9r6nqs2oxpBfC0HF63WsZVawAhhPJNI",
)
DEFAULT_LLM_BASE_URL = os.getenv("CLOSEAI_BASE_URL", "https://api.openai-proxy.org/v1")
DEFAULT_LLM_MODEL = os.getenv("CLOSEAI_MODEL", "gpt-4o-mini")


def load_hawkes_params(
    fallback: Optional[Dict[str, float]] = None,
    path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    加载 Hawkes 参数，优先使用工件文件，其次回退到默认/提供的 fallback。
    """
    base = dict(fallback or DEFAULT_HAWKES_PARAMS)
    source = path or HAWKES_PARAM_PATH
    if source.exists():
        try:
            with source.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for k in base:
                if k in data:
                    base[k] = float(data[k])
        except Exception:
            # 读取失败则静默回退默认值
            pass
    return base


def get_topics(k: Optional[int] = None) -> List[str]:
    if k is None:
        return list(DEFAULT_TOPICS)
    return DEFAULT_TOPICS[: max(k, 0)]


def get_data_dir() -> Path:
    primary = (PROJECT_ROOT / DEFAULT_DATA_DIR).resolve()
    if primary.exists():
        return primary
    fallback = (PROJECT_ROOT / FALLBACK_DATA_DIR).resolve()
    return fallback if fallback.exists() else primary


def get_real_data_path() -> Path:
    return DEFAULT_REAL_DATA_PATH
