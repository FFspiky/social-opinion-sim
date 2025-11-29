import time
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

# 跨平台字体设置，尽量兼容 Windows / macOS，支持中文显示
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from simulate import simulate_steps
from env.social_env import SocialEnv


def collect_heat_history_df(heat_history: List[dict]) -> pd.DataFrame:
    if not heat_history:
        return pd.DataFrame(columns=["time"])
    return pd.DataFrame(heat_history)


def collect_posts_df(env: SocialEnv) -> pd.DataFrame:
    rows = []
    for p in env.posts:
        rows.append({
            "time": p.time_step,
            "author": p.author,
            "sentiment": p.sentiment,
            "topic": p.topic or "未标注",
            "tag": p.tag,
            "text": p.text,
        })
    return pd.DataFrame(rows)


def collect_agent_timeline(steps: List[List], agents) -> pd.DataFrame:
    """
    汇总每个 Agent 在每个时间步的行为（post/retweet/silent），并附带情绪与话题。
    """
    rows = []
    agent_names = list(agents.keys())
    for t_idx, posts in enumerate(steps, start=1):
        for name in agent_names:
            agent_posts = [p for p in posts if p.author == name]
            if agent_posts:
                for p in agent_posts:
                    rows.append({
                        "time": t_idx,
                        "agent": name,
                        "action": "post" if p.tag != "retweet" else "retweet",
                        "sentiment": p.sentiment,
                        "topic": p.topic or "未标注",
                        "text": p.text,
                    })
            else:
                rows.append({
                    "time": t_idx,
                    "agent": name,
                    "action": "silent",
                    "sentiment": "NEUTRAL",
                    "topic": "无",
                    "text": "",
                })
    return pd.DataFrame(rows)


def render_steps(steps: List[Dict[str, Any]]):
    """将流程步骤结果以表格形式展示，成功/失败都会列出。"""
    if not steps:
        st.info("暂无流程记录")
        return
    df = pd.DataFrame(steps)
    df["状态"] = df["状态"].map(lambda s: "✅" if s == "success" else "❌")
    st.table(df[["状态", "步骤", "详情"]])


def render_network(env: SocialEnv):
    """绘制社交网络拓扑。"""
    if not hasattr(env, "G"):
        st.info("无社交网络数据")
        return
    G = env.G
    plt.figure(figsize=(6, 5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=800, font_size=9)
    st.pyplot(plt.gcf())
    plt.clf()


def render_prediction(heat_history: List[dict]):
    """简单预测：用最后两步的增量线性外推，给出 P10/P50/P90 区间。"""
    if len(heat_history) < 2:
        st.info("热度数据不足，无法预测")
        return
    df = pd.DataFrame(heat_history).set_index("time")
    last_time = df.index.max()
    future_steps = [last_time + i for i in range(1, 4)]

    fig, ax = plt.subplots(figsize=(8, 4))
    for topic in df.columns:
        if topic == "time":
            continue
        series = df[topic]
        if len(series) < 2:
            continue
        delta = series.iloc[-1] - series.iloc[-2]
        base = series.iloc[-1]
        p50 = [base + delta * i for i in range(1, 4)]
        p10 = [max(0, v * 0.8) for v in p50]
        p90 = [max(0, v * 1.2) for v in p50]
        ax.plot(df.index, series, label=f"{topic} (历史)")
        ax.fill_between(future_steps, p10, p90, alpha=0.2)
        ax.plot(future_steps, p50, linestyle="--", label=f"{topic} 预测 P50")

    ax.set_title("话题热度预测（简单外推，仅供参考）")
    ax.set_xlabel("时间步")
    ax.set_ylabel("热度")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.clf()


def compute_mape(pred_heat_df: pd.DataFrame, real_df: pd.DataFrame):
    """
    计算 MAPE：pred_heat_df 需包含 time, topic, heat_pred；real_df 需包含 time, topic, heat_real。
    返回 overall, per_topic(series), 对齐后的明细。
    """
    merged = pred_heat_df.merge(real_df, on=["time", "topic"], how="inner")
    if merged.empty:
        return None, None, None
    # 避免除零：对真实热度加一个极小值
    merged["heat_real_safe"] = merged["heat_real"].astype(float) + 1e-6
    merged["ape"] = (merged["heat_pred"] - merged["heat_real"]).abs() / merged["heat_real_safe"]
    overall = merged["ape"].mean() * 100
    per_topic = merged.groupby("topic")["ape"].mean().sort_values() * 100
    return overall, per_topic, merged


def main():
    st.set_page_config(page_title="话题热度模拟", layout="wide")
    st.title("多智能体舆论模拟：话题与热度可视化")

    # 控制面板
    st.sidebar.header("模拟参数")
    T = st.sidebar.slider("模拟时间步数", min_value=5, max_value=30, value=10, step=1)
    base_seed = st.sidebar.number_input("随机种子", min_value=0, max_value=9999, value=42)
    delay_sec = st.sidebar.slider("每步界面延迟（秒）", 0.0, 2.0, 0.2, 0.05)
    request_delay = st.sidebar.slider("API 请求间隔（秒）", 0.0, 2.0, 0.2, 0.05)
    st.sidebar.subheader("双衰减核参数（数值输入）")
    mu_fast = st.sidebar.number_input("mu_fast（爆发项）", min_value=0.0, max_value=500.0, value=1.0, step=0.1, format="%.3f")
    mu_slow = st.sidebar.number_input("mu_slow（长尾项）", min_value=0.0, max_value=500.0, value=0.5, step=0.1, format="%.3f")
    h_base = st.sidebar.number_input("H_base（外生基线）", min_value=0.0, max_value=500.0, value=5.0, step=0.5, format="%.3f")
    lambda_fast = st.sidebar.number_input("lambda_fast（短时衰减）", min_value=0.1, max_value=5.0, value=3.5, step=0.1, format="%.3f")
    lambda_slow = st.sidebar.number_input("lambda_slow（长时衰减）", min_value=0.01, max_value=2.0, value=0.5, step=0.01, format="%.3f")

    real_file = st.sidebar.file_uploader("上传真实话题热度 CSV (列: time, topic, heat)", type=["csv"])

    if "topics_input" not in st.session_state:
        st.session_state["topics_input"] = "数据安全,全运会夺冠,明星结婚"
    topics_input = st.sidebar.text_area(
        "自定义话题（逗号或换行分隔）",
        st.session_state["topics_input"],
        height=80,
        placeholder="示例：\n数据安全\n全运会夺冠\n明星结婚",
        key="topics_text",
    )
    st.session_state["topics_input"] = topics_input
    raw_topics = topics_input.replace("\n", ",")
    topics = [t.strip() for t in raw_topics.split(",") if t.strip()]

    if st.button("开始模拟"):
        steps_log: List[Dict[str, Any]] = []
        steps_log.append({"步骤": "解析话题", "状态": "success", "详情": ", ".join(topics) or "未填写"})

        env = None
        sim_result = None
        try:
            st.info("正在创建环境并运行，请稍候...")
            env, steps, heat_history = simulate_steps(
                T=T,
                seed=base_seed,
                topics=topics,
                request_delay=request_delay,
                hawkes_params={
                    "mu_fast": mu_fast,
                    "mu_slow": mu_slow,
                    "H_base": h_base,
                    "lambda_fast": lambda_fast,
                    "lambda_slow": lambda_slow,
                },
            )
            sim_result = (env, steps, heat_history)
            steps_log.append({"步骤": "运行模拟", "状态": "success", "详情": f"完成 {T} 步"})
            st.success("模拟完成")
        except Exception as e:
            steps_log.append({"步骤": "运行模拟", "状态": "fail", "详情": str(e)})
            render_steps(steps_log)
            st.error(f"模拟失败：{e}")
            return

        if not sim_result:
            render_steps(steps_log)
            return

        env, steps, heat_history = sim_result

        # 话题热度折线图
        if heat_history:
            try:
                heat_df = collect_heat_history_df(heat_history).set_index("time")
                st.subheader("话题热度随时间变化")
                st.line_chart(heat_df)
                steps_log.append({"步骤": "绘制话题热度", "状态": "success", "详情": "已生成折线图"})
            except Exception as e:
                steps_log.append({"步骤": "绘制话题热度", "状态": "fail", "详情": str(e)})
        else:
            st.info("当前未配置话题或无热度数据。")
            steps_log.append({"步骤": "绘制话题热度", "状态": "fail", "详情": "无数据"})

        # 话题热度预测
        st.subheader("话题热度预测")
        try:
            render_prediction(heat_history)
            steps_log.append({"步骤": "绘制预测", "状态": "success", "详情": "简单外推"})
        except Exception as e:
            steps_log.append({"步骤": "绘制预测", "状态": "fail", "详情": str(e)})

        # 时间步帖子展示
        try:
            st.subheader("按时间步的事件内容")
            for t_idx, posts in enumerate(steps, start=1):
                with st.expander(f"时间步 {t_idx} ({len(posts)} 条)"):
                    if posts:
                        rows = [{
                            "作者": p.author,
                            "情绪": p.sentiment,
                            "话题": p.topic or "未标注",
                            "标签": p.tag,
                            "内容": p.text,
                        } for p in posts]
                        st.table(pd.DataFrame(rows))
                    else:
                        st.write("无新帖子")
                    if delay_sec > 0 and t_idx < len(steps):
                        time.sleep(delay_sec)
            steps_log.append({"步骤": "展示事件内容", "状态": "success", "详情": "按时间步展开"})
        except Exception as e:
            steps_log.append({"步骤": "展示事件内容", "状态": "fail", "详情": str(e)})

        # Agent 行为时间线
        try:
            st.subheader("Agent 行为时间线")
            agent_timeline = collect_agent_timeline(steps, env.agents)
            st.dataframe(agent_timeline)
            steps_log.append({"步骤": "展示行为时间线", "状态": "success", "详情": "表格输出"})
        except Exception as e:
            steps_log.append({"步骤": "展示行为时间线", "状态": "fail", "详情": str(e)})

        # 社交网络图
        try:
            st.subheader("社交网络拓扑")
            render_network(env)
            steps_log.append({"步骤": "绘制网络", "状态": "success", "详情": "spring_layout"})
        except Exception as e:
            steps_log.append({"步骤": "绘制网络", "状态": "fail", "详情": str(e)})

        # 真实数据对齐与 MAPE
        if real_file is not None:
            try:
                real_df_raw = pd.read_csv(real_file)
                if not set(["time", "topic", "heat"]).issubset(real_df_raw.columns):
                    raise ValueError("CSV 需包含列: time, topic, heat")
                real_df = real_df_raw.rename(columns={"heat": "heat_real"})
                if heat_history:
                    heat_df_reset = collect_heat_history_df(heat_history)
                    pred_long = heat_df_reset.melt(id_vars="time", var_name="topic", value_name="heat_pred")
                    overall, per_topic, merged = compute_mape(pred_long, real_df)
                    st.subheader("MAPE 校准")
                    if overall is None:
                        st.info("无可对齐的数据，无法计算 MAPE")
                        steps_log.append({"步骤": "计算 MAPE", "状态": "fail", "详情": "无对齐数据"})
                    else:
                        st.write(f"总体 MAPE: {overall:.2f}%")
                        st.write("各话题 MAPE：")
                        st.table(per_topic.reset_index().rename(columns={"ape": "MAPE%"}))
                        st.dataframe(merged)
                        steps_log.append({"步骤": "计算 MAPE", "状态": "success", "详情": f"总体 {overall:.2f}%"})
                else:
                    st.info("无预测热度，无法计算 MAPE")
                    steps_log.append({"步骤": "计算 MAPE", "状态": "fail", "详情": "无预测热度"})
            except Exception as e:
                st.error(f"MAPE 计算失败：{e}")
                steps_log.append({"步骤": "计算 MAPE", "状态": "fail", "详情": str(e)})

        # 汇总所有帖子表
        try:
            st.subheader("全部帖子汇总")
            posts_df = collect_posts_df(env)
            st.dataframe(posts_df)
            steps_log.append({"步骤": "汇总帖子", "状态": "success", "详情": f"共 {len(posts_df)} 条"})
        except Exception as e:
            steps_log.append({"步骤": "汇总帖子", "状态": "fail", "详情": str(e)})

        st.subheader("流程结果")
        render_steps(steps_log)


if __name__ == "__main__":
    main()
