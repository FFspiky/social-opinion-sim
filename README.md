# 多智能体舆论博弈模拟（话题热度预测版）

基于多代理 + Hawkes 双衰减核（爆发/长尾）的舆论演化实验，提供训练、模拟、前端对比。

## 最新要点
- 5 个代表性微博画像：权威媒体/官方、KOL、杠精、护卫队、吃瓜群众，默认话题为最新训练集前 5 个（哈工大你玩真的啊、为啥网上的药比实体药店更便宜、晒晒家乡隐藏款土特产、太空发快递可以当日达了、春晚节目）。
- 最新 Hawkes 最优参数（未归一化）：mu_fast=0.58896, mu_slow=0.19392, H_base=1.81348, lambda_fast=4.99670, lambda_slow=0.66370；默认 heat_scale=1e4 便于对齐真实量级。
- 模拟支持权重调度：爆发角色用 lambda_fast 衰减，长尾角色用 lambda_slow，按热度和上次发声时间决定出场概率（约 40%/步）。
- 前端可一键加载默认真实数据（`../huoju/dataset_peak350/classified_events_35_2024Q1-Q4_peak350_v2.csv`），自动对齐时间步并计算 MSE/MAPE，绘制 Sim vs Real 对比。

## 功能概览
- 多角色 Agent：5 画像（official_media / kol / troll / defender / crowd），话题注意力分配与记忆流。
- 话题热度：TopicManager 记录帖子与热度，双衰减核（mu_fast/mu_slow/H_base/lambda_fast/lambda_slow）+ 可调 heat_scale。
- 前端：自定义话题与时间步（默认 350），展示热度曲线、帖子列表、Agent 时间线；支持真实数据对比（MAPE/MSE）。

## 数据
- 默认真实/训练参考：`../huoju/dataset_peak350/classified_events_35_2024Q1-Q4_peak350_v2.csv`（topic, heat, timestamp，约 350 步/话题）。
- 旧示例：`datasets/`、`datasets_huoju*`、`classified_events_35.csv` 等。

## 依赖与环境
```bash
python -m venv venv
# PowerShell: .\venv\Scripts\Activate.ps1
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## 训练（双衰减核）
```bash
python train.py --data_dir datasets_huoju_norm
python train.py --data_dir datasets_huoju_norm --random_test --seed 123
```
模型形式：
```
pred_t = H_base + mu_fast*M_fast + mu_slow*M_slow
M_fast = M_fast*exp(-lambda_fast) + y_{t-1}
M_slow = M_slow*exp(-lambda_slow) + y_{t-1}
约束：lambda_fast > lambda_slow > 0
```

## 模拟与对比
- 命令行单次模拟并对比默认真实数据：
```bash
python simulate.py   # __main__ 调用 run_and_compare，T=350，默认话题与参数
```
- Streamlit 前端：
```bash
streamlit run app_streamlit.py
# 浏览器访问 http://localhost:8501
```
侧边栏：时间步、随机种子、默认话题（可自定义），可选“使用默认真实数据”，也可上传 CSV（time/topic/heat 或 timestamp/topic/heat）。自动计算总体/分话题 MAPE、MSE 并绘图。

## 主要文件
- `simulate.py`：默认最优参数、权重调度、run_and_compare（Sim vs Real 图与 MSE/MAPE）。
- `app_streamlit.py`：前端交互，默认加载最新真实数据，可上传替换。
- `agents/agent.py`：角色提示/行为决策；`env/social_env.py`：热度管理、Agent 出场权重调度。
- `train.py` / `utils/data_loader.py`：训练加载与参数拟合；`utils/spread_model.py`：双衰减核预测。

## 注意
- 需要可用的 LLM API Key（见 `agents/llm_client.py`）；受限网络请自配代理/离线策略。
- heat_scale 可按真实量级调节；权重调度可在 `env/social_env.py` 进一步微调爆发/长尾参与度。
