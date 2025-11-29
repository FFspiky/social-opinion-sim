import random
import numpy as np
import matplotlib.pyplot as plt

from utils.data_loader import load_all_datasets
from utils.spread_model import HawkesPredictor


# 可硬编码最优参数（需在运行 train.py 后更新）
BEST_PARAMS = np.array([0.8505534125, 0.0862380168, 4.9818004704, 4.4057477813, 1.9738589443], dtype=float)


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    safe_true = np.maximum(y_true, 1e-6)
    return float(np.mean(np.abs(y_pred - safe_true) / safe_true) * 100)


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_pred - y_true
    return float(np.mean(diff * diff))


def rollout(params: np.ndarray, series: np.ndarray) -> np.ndarray:
    return HawkesPredictor.predict_dual_decay(params, series)


def main():
    # 字体设置，兼容 mac 显示中文
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    events = load_all_datasets("datasets")

    metrics = []
    for ev in events:
        test = ev["test"]
        # teacher forcing 滚动预测
        pred_full = rollout(BEST_PARAMS, test)
        mse = compute_mse(test, pred_full)
        mape = compute_mape(test, pred_full)
        metrics.append((ev["name"], mse, mape, test, pred_full))

    avg_mape = np.mean([mape for _, _, mape, _, _ in metrics])
    avg_mse = np.mean([mse for _, mse, _, _, _ in metrics])
    print(f"Average MSE on held-out Test: {avg_mse:.2f}")
    print(f"Average MAPE on held-out Test: {avg_mape:.2f}%")
    if avg_mape < 20.0:
        print("Model Validation Success!")

    # 随机抽 4 个事件可视化
    sample = random.sample(metrics, k=4) if len(metrics) >= 4 else metrics
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for ax, (name, mse, mape, y_true, y_pred) in zip(axes, sample):
        ax.plot(y_true, color="black", label="真实")
        ax.plot(y_pred, color="red", linestyle="--", label="预测")
        ax.set_title(f"{name} - MSE {mse:.1f} / MAPE {mape:.2f}%")
        ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
