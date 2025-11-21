# 多智能体社会舆情模拟

一个基于大模型的多智能体舆情博弈小实验，用不同公关策略对品牌危机进行模拟，并提供命令行与 Streamlit 前端展示。

## 功能亮点
- 预置 3 种公关策略：`S0` 不作为、`S1` 延迟道歉、`S2` 快速澄清+定期更新。
- 社交图 + 角色画像驱动的 Agent，通过 LLM 决策是否发帖、回复以及情绪倾向。
- 内置记忆流（向量召回与反思），让 Agent 的发帖更连贯。
- Streamlit 前端可视化帖子列表与时间序列负面占比。

## 目录结构
- `simulate.py`：命令行入口，单次运行指定策略。
- `app_streamlit.py`：前端入口，交互式选择策略与参数。
- `agents/agent.py`：核心 Agent 逻辑（记忆、决策、发帖）。
- `agents/llm_client.py`：封装 SiliconFlow DeepSeek 调用。
- `agents/memory.py`：记忆流与反思。
- `env/social_env.py`：社交网络环境与时间步推进。
- `pr_strategies/strategies.py`：公关策略实现。
- `run_simulation.cmd`：Windows 一键启动（激活 venv + 运行 Streamlit）。
- `requirements.txt`：依赖列表。

## 环境准备
1. **Python 3.10+**（推荐 3.11/3.12）。
2. 安装依赖：
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows 使用 venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. 配置大模型密钥（SiliconFlow DeepSeek-Terminus）：
   在项目根目录创建 `.env`，内容示例：
   ```env
   SILICONFLOW_API_KEY=你的APIKey
   SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
   SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V3.1-Terminus
   ```
   首次运行会自动下载 `sentence-transformers` 模型，需联网。

## 运行方式
### 1) 命令行单次模拟
```bash
python simulate.py           # 默认 S1，T=8，seed=123
# 或指定参数（在 simulate.py 内修改默认调用，或在其他脚本中使用 run_once）
```
运行过程中会打印每个时间步的发帖，结束后返回包含所有帖子的数据框（用于分析或在前端展示）。

### 2) Streamlit 前端
```bash
streamlit run app_streamlit.py
# 浏览器打开 http://localhost:8501 交互选择策略、时间步、随机种子
```
Windows 用户也可以双击 `run_simulation.cmd` 自动激活 venv 并启动前端。

## 参数与扩展
- 策略选择：`S0`/`S1`/`S2`，见 `pr_strategies/strategies.py`。
- Agent 数量与角色：在 `simulate.py` 的 `build_agents` 可增删角色或调整 speak 概率。
- 社交图生成：`build_graph` 中可自定义关注关系生成逻辑。
- 记忆与反思：`agents/memory.py` 中可调整反思阈值、半衰期、嵌入模型。
- 如需并行比较多种策略，可调用 `create_simulation_instance` 创建多套环境与策略。

## 注意事项
- 本项目调用云端 LLM，会产生少量调用费用与网络时延；请确保 `.env` 已配置有效密钥。
- 运行 Streamlit/CLI 时，若提示模型或依赖下载失败，请检查网络或预先离线下载相关模型文件。
- 代码中未包含真实业务数据，只用于 Demo/教学目的，可按需定制提示词和角色画像。
