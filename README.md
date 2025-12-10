# 多智能体舆论博弈模拟（话题热度预测版）

基于多角色 + Hawkes 双衰减核（爆发/长尾）的舆论演化实验，提供训练、模拟、前端可视化与真实数据对比。

## 核心更新
- **双驱动 Agent**：`agents/agent.py` 引入心理参数、信任矩阵和 driver_mode（brain/reflex）；新增风格模板 `config/styles.py`，按角色注入微博话语风格。
- **环境动力学**：`env/social_env.py` 按 Hawkes 强度与热度实时计算 phase 和 global_tension，并下发 env_context 影响决策，防止全局沉默时自动回退转发。
- **初始播种**：`simulate.py::inject_initial_rumor` 使用话题相关模板随机生成首条爆料，无需 LLM。
- **全局优化回归**：`train.py` 支持 CMA-ES 粗搜 + 多起点 L-BFGS-B 精调（`--use_global_init`），保留多初始 L-BFGS-B。
- **配置集中**：`config/settings.py` 汇总默认话题、数据目录、真实数据路径、Hawkes/LLM 默认参数；`load_hawkes_params` 统一读取 `artifacts/hawkes_params.json`。
- **依赖**：`requirements.txt` 含 `cma>=3.2.2` 等。

## 环境准备
```bash
python -m venv venv
# macOS/Linux: source venv/bin/activate
# PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 配置与数据
- **默认话题**：见 `config/settings.py::DEFAULT_TOPICS`（前 5 个中文话题）。
- **数据目录**：`DATA_DIR` 环境变量或默认 `datasets_huoju_norm`，若不存在回退 `dataset_peak350`。
- **真实数据路径**：`config.settings.DEFAULT_REAL_DATA_PATH`（默认指向 `../huoju/dataset_peak350/classified_events_35_2024Q1-Q4_peak350_v2.csv`）。
- **Hawkes 参数来源**：优先 `artifacts/hawkes_params.json`，否则使用配置内默认值。
- **LLM 配置**：默认基址 `https://api.openai-proxy.org/v1`，可用环境变量 `CLOSEAI_API_KEY/CLOSEAI_BASE_URL/CLOSEAI_MODEL` 覆盖；未配置/连不上时 brain 模式角色可能沉默。

## 训练 Hawkes 参数
双衰减核形式：
```
pred_t = H_base + mu_fast*M_fast + mu_slow*M_slow
M_fast = M_fast*exp(-lambda_fast) + y_{t-1}
M_slow = M_slow*exp(-lambda_slow) + y_{t-1}
约束：lambda_fast > lambda_slow > 0
```

- 基本命令（默认数据目录 `dataset_peak350`，按时间 80/10/10 切分，归一化开启）：
```bash
python train.py
```
- 旧流程：仅多初始 L-BFGS-B
```bash
python train.py --lbfgs_maxiter 300
```
- 新流程：CMA-ES 全局搜索 + 多起点 L-BFGS-B 精调
```bash
python train.py \
  --use_global_init \
  --global_n_starts 30 \
  --cma_maxiter 40 \
  --cma_popsize 16 \
  --cma_sigma0 0.3 \
  --perturb_scale 0.15 \
  --lbfgs_maxiter 300 \
  --seed 42
```
- 常用参数：`--data_dir` 指定数据目录；`--random_test` 随机抽 10% 作为测试集；`--seed` 控制切分/扰动。
- 归一化默认开启，参数 bounds（归一化场景）：
  - `mu_fast, mu_slow ∈ [1e-6, 500]`
  - `H_base ∈ [1e-6, 100]`（如关闭归一化或数据量级大，需手动放宽）
  - `lambda_fast ∈ [0.5, 5.0]`
  - `lambda_slow ∈ [0.01, 2.0]`
  - 强制 `lambda_fast > lambda_slow`
- 输出：控制台打印最优参数及 Train/Val/Test 的 MSE/MAPE。默认不会自动写文件；如需持久化，请将输出参数写入 `artifacts/hawkes_params.json` 以供模拟/前端使用。

## 模拟与真实对比
- 命令行快速对比（默认 5 个话题、350 步，使用最新工件参数与真实数据，生成 `simulation_vs_real.png`）：
```bash
python simulate.py
```
- 主要逻辑：`simulate.py::run_and_compare` 调用 `simulate_steps` 生成模拟热度，按真实数据对齐后计算整体/分话题 MSE、MAPE 并绘图。`simulate_steps` 会为每个话题注入首条爆料，环境按 Hawkes 强度/热度计算 phase 与 global_tension，并传入 Agent。

## 交互前端
```bash
streamlit run app_streamlit.py
# 浏览器访问 http://localhost:8501
```
- 侧边栏可配置时间步、随机种子、话题、自定义/默认真实数据（time/topic/heat 或 timestamp/topic/heat）。
- 页面展示：热度曲线、时间步帖子列表、Agent 行为时间线、Sim vs Real 指标与可视化。

## 主要模块
- `config/settings.py`：默认话题、数据路径、Hawkes/LLM 默认参数，`load_hawkes_params()`、`get_data_dir()`、`get_real_data_path()`。
- `config/styles.py`：按角色定义微博风格提示，Agent 构造 system prompt 时注入。
- `train.py`：CMA-ES + 多起点 L-BFGS-B 拟合双衰减核（可选旧流程）。
- `utils/data_loader.py`：CSV 加载与 80/10/10 切分；normalize=True 时返回 `scale`。
- `utils/spread_model.py`：双衰减核预测。
- `simulate.py`：多 Agent 仿真，种子爆料、热度曲线、与真实数据对比。
- `env/social_env.py`：Hawkes 记忆与热度、phase/global_tension、出场权重调度、env_context 下发。
- `agents/agent.py`：双驱动（brain/reflex）决策、心理/信任参数、风格化 prompt；`agents/llm_client.py`：LLM 调用封装。

## 调参建议
- 若全局搜索不稳定：调大 `cma_sigma0`（如 0.5）、`perturb_scale`（如 0.3）、`cma_popsize`（如 32）、`global_n_starts`（如 50）、`cma_maxiter`（如 60）。
- 若 L-BFGS-B 收敛浅：`--lbfgs_maxiter` 提高到 500。
- 关注 MAPE 可在 loss 中加入相对误差权重（需改代码）；仅调优化器未必提升 MAPE。

## 注意事项
- 确保数据目录存在且包含 CSV，否则加载会提示并返回空列表。
