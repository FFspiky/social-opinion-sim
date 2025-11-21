# app_streamlit.py

import time
import random
import streamlit as st
import pandas as pd

from simulate import create_simulation_instance
from agents.llm_client import LLMClient



STRATEGY_CONFIGS = [
    ("S0", "不作为（基线）"),
    ("S1", "延迟道歉"),
    ("S2", "快速澄清+进展更新"),
]

def summarize_strategy_for_llm(inst):
    """
    将某个策略的 env 和 posts 压缩成一段给大模型看的摘要文本。
    inst: {"code", "name", "env", "strategy", "all_posts"}
    返回: 字符串
    """
    env = inst["env"]
    code = inst["code"]
    name = inst["name"]
    posts = env.posts

    stats_df = collect_sentiment_stats(posts)
    if stats_df.empty:
        return f"策略 {code} - {name}：没有产生任何帖子。\n"

    final_neg = stats_df["neg_ratio"].iloc[-1]
    peak_neg = stats_df["neg_ratio"].max()
    peak_time = int(stats_df.loc[stats_df["neg_ratio"].idxmax(), "time"])
    avg_neg = stats_df["neg_ratio"].mean()

    # 选几条代表性帖子：前几条负面、正面和官方声明
    negative_posts = [p for p in posts if p.sentiment == "NEGATIVE"]
    positive_posts = [p for p in posts if p.sentiment == "POSITIVE"]
    official_posts = [p for p in posts if p.tag == "official"]

    def sample_text(post_list, max_count=2):
        samples = []
        for p in post_list[:max_count]:
            content = p.text
            if len(content) > 120:
                content = content[:120] + "..."
            samples.append(
                f"[t={p.time_step}] {p.author} ({p.sentiment}): {content}"
            )
        return samples

    neg_samples = sample_text(negative_posts, max_count=3)
    pos_samples = sample_text(positive_posts, max_count=3)
    off_samples = sample_text(official_posts, max_count=3)

    lines = []
    lines.append(f"策略 {code} - {name}:")
    lines.append(f"- 最终负面比例: {final_neg:.2f}")
    lines.append(f"- 峰值负面比例: {peak_neg:.2f}，出现在时间步 t={peak_time}")
    lines.append(f"- 平均负面比例: {avg_neg:.2f}")
    lines.append("")
    if off_samples:
        lines.append("【代表性的官方声明】")
        lines.extend(f"  - {s}" for s in off_samples)
    if neg_samples:
        lines.append("【代表性的负面发言】")
        lines.extend(f"  - {s}" for s in neg_samples)
    if pos_samples:
        lines.append("【代表性的正面/支持发言】")
        lines.extend(f"  - {s}" for s in pos_samples)

    lines.append("\n")
    return "\n".join(lines)


def collect_sentiment_stats(posts):
    """
    从 env.posts 中计算每个时间步的负面比例。
    posts: List[Post]
    返回 DataFrame: time, total, neg, neg_ratio
    """
    rows = []
    for p in posts:
        rows.append({
            "time": p.time_step,
            "sentiment": p.sentiment,
            "text": p.text,
        })
    if not rows:
        return pd.DataFrame(columns=["time", "total", "neg", "neg_ratio"])

    df = pd.DataFrame(rows)
    df["is_negative"] = df["sentiment"] == "NEGATIVE"
    grouped = df.groupby("time").agg(
        total=("text", "count"),
        neg=("is_negative", "sum"),
    )
    grouped["neg_ratio"] = grouped["neg"] / grouped["total"]
    grouped = grouped.reset_index()

    return grouped


def main():
    st.set_page_config(page_title="多策略舆情博弈模拟", layout="wide")
    st.title("多智能体舆论模拟 · 三种公关策略对比（流式展示）")

    # 控制区域
    st.sidebar.header("模拟参数")
    T = st.sidebar.slider("模拟时间步数", min_value=5, max_value=30, value=10, step=1)
    base_seed = st.sidebar.number_input("随机种子（保证三种策略环境一致）", min_value=0, max_value=9999, value=42)

    delay_sec = st.sidebar.slider("每个时间步之间的延迟（秒）", 0.0, 2.0, 0.5, 0.1)

    st.sidebar.markdown("当前策略组合：")
    for code, name in STRATEGY_CONFIGS:
        st.sidebar.write(f"- {code}：{name}")

    if st.button("开始三策略并行模拟"):
        status_placeholder = st.empty()
        status_placeholder.write("⌛ 正在创建三套模拟环境（可能会调用多次大模型，请稍等片刻）...")

        instances = []
        for idx, (code, name) in enumerate(STRATEGY_CONFIGS):
            status_placeholder.write(f"⌛ 正在创建第 {idx+1} 套环境：{code} - {name} ...")
            env, strategy = create_simulation_instance(code, seed=base_seed)
            instances.append({
                "code": code,
                "name": name,
                "env": env,
                "strategy": strategy,
                "all_posts": [],
                "history_lines": [],
            })

        status_placeholder.success("✅ 三套环境创建完毕，开始逐时间步模拟！")

        # 三列布局：每列对应一种策略
        cols = st.columns(3)

        # 为每列准备占位符：一个显示最新帖子，一个显示曲线
        col_placeholders = []
        for col, (code, name) in zip(cols, STRATEGY_CONFIGS):
            with col:
                st.subheader(f"{code} - {name}")
                chart_placeholder = st.empty()
                text_placeholder = st.empty()
                col_placeholders.append({
                    "chart": chart_placeholder,
                    "text": text_placeholder,
                })

        # 按时间步推进
        for t in range(1, T + 1):
            st.markdown(f"### 时间步 {t}")

            for idx, inst in enumerate(instances):
                env = inst["env"]
                strategy = inst["strategy"]

                # 推进一步
                new_posts = env.step(pr_strategy=strategy)
                inst["all_posts"].extend(new_posts)

                latest_texts = []
                step_lines = []  # 本时间步所有行，带 t 信息
                for p in new_posts:
                    content = p.text
                    if len(content) > 120:
                        content = content[:120] + "..."
                    line = f"[t={p.time_step}] **{p.author}** ({p.sentiment}): {content}"
                    latest_texts.append(line)
                    step_lines.append(line)

                # 把本时间步的内容加入历史
                inst["history_lines"].append(f"### 时间步 {t}")
                inst["history_lines"].extend(step_lines)

                with cols[idx]:
                    # 最新发言（只显示这一步）
                    if latest_texts:
                        col_placeholders[idx]["text"].markdown(
                            "#### 最新发言\n" + "\n\n".join(latest_texts)
                        )
                    else:
                        col_placeholders[idx]["text"].markdown("#### 最新发言\n（本步无新发言）")

                    # 完整历史记录（放在一个折叠框里）
                    with st.expander("查看该策略的全部发言记录", expanded=False):
                        if inst["history_lines"]:
                            st.markdown("\n\n".join(inst["history_lines"]))
                        else:
                            st.write("暂无发言记录。")

                # 计算并更新负面比例曲线
                stats_df = collect_sentiment_stats(env.posts)
                if not stats_df.empty:
                    chart_data = stats_df.set_index("time")[["neg_ratio"]]
                    col_placeholders[idx]["chart"].line_chart(chart_data)

            # 在前端做一点延迟，让“流式”效果更明显
            if delay_sec > 0:
                time.sleep(delay_sec)

        st.success("模拟结束 ✅ 你可以调整参数重新运行。")
        # ---------- 模拟结束后：调用大模型思考模式，写总结报告 ---------- #

        st.subheader("🧠 大模型总结报告（思考模式）")

        # 1. 组装给大模型看的摘要文本
        summary_sections = []
        for inst in instances:
            section_text = summarize_strategy_for_llm(inst)
            summary_sections.append(section_text)

        summary_text = "\n\n".join(summary_sections)

        # 2. 构造 prompt
        system_prompt = (
            "你是一名精通危机公关与舆论传播分析的专家。"
            "你会根据给出的三种公关策略模拟结果，分析舆情走向、策略得失，并给出建议。"
        )

        user_prompt = f"""
下面是三种不同公关策略在社交媒体舆情模拟中的结果摘要。
每个策略都包含：最终负面比例、峰值负面比例、代表性发言样本。

请你在“思考模式”下认真分析，并输出一份结构化的中文总结报告，内容包括：

1. 三种策略的舆情走势对比（哪些更容易压制负面扩散，哪些会加剧对立）。
2. 从代表性发言中可以看出用户情绪和信任感的哪些变化。
3. 每种策略的优点 / 缺点。
4. 综合来说，在类似的数据泄露危机场景中，哪种策略更优，是否可以组合使用。
5. 给出面向企业公关团队的 3~5 条可执行建议。

以下是三种策略的结果摘要：

{summary_text}
"""

        # 3. 调用思考模式的 chat_thinking（内部会启用 enable_thinking=True）
        try:
            llm = LLMClient()
            report = llm.chat_thinking(system_prompt, user_prompt)
            st.markdown(report)
        except Exception as e:
            st.error(f"生成总结报告时出错：{e}")

if __name__ == "__main__":
    main()
