# simulate.py
from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Optional

import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import (
    DEFAULT_REAL_DATA_PATH,
    DEFAULT_SIM_SEED,
    DEFAULT_TOPICS,
    load_hawkes_params,
)
# 把项目根目录加入模块搜索路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from agents.agent import Agent
from agents.llm_client import LLMClient
from env.social_env import SocialEnv
import math
import matplotlib.pyplot as plt
# 非随机全量训练得到的最优参数（归一化数据），优先读取工件
BEST_HAWKES_PARAMS = load_hawkes_params()


def pick_default_topics(seed: int = DEFAULT_SIM_SEED, k: int = 5) -> List[str]:
    """
    直接使用 DEFAULT_TOPICS，保证确定性。
    """
    return DEFAULT_TOPICS[: min(k, len(DEFAULT_TOPICS))]


def build_agents(llm: LLMClient, topics: Optional[List[str]] = None) -> Dict[str, Agent]:
    """构建 5 个代表性画像的微博生态 Agent。"""
    agents: Dict[str, Agent] = {}

    agents["BrandOfficial"] = Agent(
        name="BrandOfficial",
        role="BrandOfficial",
        profile="权威媒体/官方：客观、冷静、强调核实，不信谣不传谣。",
        llm_client=llm,
        topics=topics,
    )

    agents["KOL"] = Agent(
        name="KOL",
        role="KOL",
        profile="意见领袖/营销号：爱带节奏，反问、悬念、情绪化吸引流量。",
        llm_client=llm,
        topics=topics,
    )

    agents["Troll"] = Agent(
        name="Troll",
        role="Troll",
        profile="极端情绪/杠精：尖锐、嘲讽、挑衅，擅长制造冲突。",
        llm_client=llm,
        topics=topics,
    )

    agents["Defender"] = Agent(
        name="Defender",
        role="Defender",
        profile="死忠粉/护卫队：极度护短，控评、呼吁理性，抵制黑子。",
        llm_client=llm,
        topics=topics,
    )

    agents["Crowd"] = Agent(
        name="Crowd",
        role="Crowd",
        profile="吃瓜群众：路人心态，简短跟风，热度高时才出现。",
        llm_client=llm,
        topics=topics,
    )

    return agents


def build_graph(agent_names):
    G = nx.DiGraph()
    G.add_nodes_from(agent_names)

    # 所有人都关注官方与KOL
    for name in agent_names:
        if name != "BrandOfficial":
            G.add_edge(name, "BrandOfficial")
        if name != "KOL":
            G.add_edge(name, "KOL")

    # 再随机补充一些关注关系
    names = list(agent_names)
    for src in names:
        for _ in range(3):
            dst = random.choice(names)
            if dst != src and not G.has_edge(src, dst):
                G.add_edge(src, dst)

    return G


def inject_initial_rumor(env: SocialEnv, topic: str | None = None):
    """
    t=0，媒体发第一条热点爆料（无需 LLM，按话题生成简短引子）。
    """
    base_topic = topic or "某热点事件"
    templates = [
        f"【最新爆料】{base_topic} 疑似有反转，细节还在核实，大家怎么看？",
        f"听说 {base_topic} 又有新瓜，真假未证，先吃瓜围观。",
        f"{base_topic} 刷屏了，内部人士称情况复杂，等官方通报？",
        f"{base_topic} 有人爆料存在风险，暂未证实，理性围观。",
    ]
    text = random.choice(templates)

    env._add_post(
        author="Media1",
        text=text,
        sentiment="NEGATIVE",
        tag="rumor",
        target_post_id=None,
        topic=topic,
    )


def simulate_steps(
    T: int = 100,
    seed: int = DEFAULT_SIM_SEED,
    topics: Optional[List[str]] = None,
    request_delay: float = 0.0,
    hawkes_params: Optional[dict] = None,
):
    """
    运行多时间步模拟，返回环境、每步新增帖子列表、以及话题热度快照。
    """
    random.seed(seed)
    if not topics:
        topics = pick_default_topics(seed=seed, k=5)

    llm = LLMClient()
    agents = build_agents(llm, topics=topics)
    G = build_graph(agents.keys())
    env = SocialEnv(agents, G, topics=topics, hawkes_params=hawkes_params or BEST_HAWKES_PARAMS)

    # 初始爆料：为每个话题种子一条（若未提供话题，则发一条默认）
    if topics:
        for tp in topics:
            inject_initial_rumor(env, topic=tp)
    else:
        inject_initial_rumor(env, topic=None)

    steps = []
    heat_history = []
    for _ in range(1, T + 1):
        new_posts = env.step(pr_strategy=None, request_delay=request_delay)
        steps.append(new_posts)
        if env.topic_manager:
            snapshot = {"time": env.t}
            for topic in env.topic_manager.topics:
                snapshot[topic] = env.topic_manager.get_heat(topic)
            heat_history.append(snapshot)
    return env, steps, heat_history


# ---------------------------------------------------------------------
# 真实数据对比工具
# ---------------------------------------------------------------------

def _load_real_series(real_path: Path, topics: List[str], max_steps: int = 350) -> Dict[str, List[float]]:
    """
    从真实数据 CSV 读取指定 topic 的热度序列，按 timestamp 排序，截断到 max_steps。
    CSV 列：topic, heat, timestamp
    """
    if not real_path.exists():
        return {t: [] for t in topics}
    df = pd.read_csv(real_path)
    series: Dict[str, List[float]] = {t: [] for t in topics}
    for t in topics:
        sub = df[df["topic"] == t].sort_values("timestamp")
        heats = sub["heat"].astype(float).tolist()[:max_steps]
        series[t] = heats
    return series


def _compute_metrics(sim_series: Dict[str, List[float]], real_series: Dict[str, List[float]]):
    """
    计算 MSE / MAPE（按公共长度对齐）。
    返回 overall 与 per-topic 结果。
    """
    def mse(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) / max(len(a), 1)

    def mape(a, b):
        return sum(abs(x - y) / max(abs(y), 1e-6) for x, y in zip(a, b)) / max(len(a), 1) * 100.0

    per_topic = {}
    mse_total = 0.0
    mape_total = 0.0
    n = 0
    for t, sim in sim_series.items():
        real = real_series.get(t, [])
        L = min(len(sim), len(real))
        if L == 0:
            continue
        sim_cut = sim[:L]
        real_cut = real[:L]
        m = mse(sim_cut, real_cut)
        p = mape(sim_cut, real_cut)
        per_topic[t] = {"mse": m, "mape": p, "len": L}
        mse_total += m
        mape_total += p
        n += 1
    overall = {"avg_mse": mse_total / n if n else 0.0, "avg_mape": mape_total / n if n else 0.0}
    return overall, per_topic


def _plot_comparison(sim_series: Dict[str, List[float]], real_series: Dict[str, List[float]], out_path: Path):
    topics = list(sim_series.keys())
    if not topics:
        return
    cols = 2
    rows = math.ceil(len(topics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), sharex=False)
    axes = axes.flatten() if isinstance(axes, (list, tuple, np.ndarray)) else [axes]
    for ax in axes[len(topics):]:
        ax.axis("off")
    for idx, topic in enumerate(topics):
        ax = axes[idx]
        sim = sim_series.get(topic, [])
        real = real_series.get(topic, [])
        L = min(len(sim), len(real))
        x_sim = list(range(len(sim)))
        ax.plot(x_sim, sim, label="Sim", color="#1f77b4", linewidth=1.2)
        ax.plot(list(range(L)), real[:L], label="Real", color="#d62728", linestyle="--", linewidth=1.0)
        ax.set_title(topic, fontsize=9)
        ax.tick_params(labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("Simulation vs Real Heat", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_and_compare(
    T: int = 350,
    seed: int = DEFAULT_SIM_SEED,
    topics: Optional[List[str]] = None,
    hawkes_params: Optional[dict] = None,
    real_data_path: Optional[Path] = None,
):
    """
    运行模拟并与最新训练数据对比，输出 MSE/MAPE 和对比图。
    """
    topics = topics or DEFAULT_TOPICS
    real_path = real_data_path or DEFAULT_REAL_DATA_PATH
    env, steps, heat_history = simulate_steps(
        T=T,
        seed=seed,
        topics=topics,
        hawkes_params=hawkes_params or BEST_HAWKES_PARAMS,
    )
    # 收集模拟热度
    sim_series = {t: [] for t in topics}
    for snap in heat_history:
        for t in topics:
            sim_series[t].append(snap.get(t, 0.0))

    # 读取真实热度
    real_series = _load_real_series(real_path, topics, max_steps=len(heat_history))

    overall, per_topic = _compute_metrics(sim_series, real_series)
    plot_path = Path("simulation_vs_real.png")
    _plot_comparison(sim_series, real_series, plot_path)

    print(f"Real data path: {real_path}")
    print(f"Simulation vs Real -> AVG MSE: {overall['avg_mse']:.4f}, AVG MAPE: {overall['avg_mape']:.2f}%")
    for t, m in per_topic.items():
        print(f"  {t}: MSE={m['mse']:.4f}, MAPE={m['mape']:.2f}%, len={m['len']}")
    print(f"Plot saved to {plot_path}")
    return overall, per_topic, plot_path


if __name__ == "__main__":
    run_and_compare(T=350, seed=123)
