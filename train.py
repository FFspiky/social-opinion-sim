"""
核心训练脚本：全局拟合 34 个事件的共同参数 [alpha, beta, mu, lambda]。
遵循 train_global_params.py 的标准：以 train+val 平均 MSE 为目标，test 仅用于报告。
"""

import math
import numpy as np
from scipy.optimize import minimize

from utils.data_loader import load_all_datasets
from utils.spread_model import HawkesPredictor


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_pred - y_true
    return float(np.mean(diff * diff))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    safe_true = np.maximum(y_true, 1e-6)
    return float(np.mean(np.abs(y_pred - safe_true) / safe_true) * 100.0)


def rollout(params: np.ndarray, series: np.ndarray) -> np.ndarray:
    """
    双衰减核：pred = H_base + mu_fast * M_fast + mu_slow * M_slow
    M_fast = M_fast * exp(-lambda_fast) + y_{t-1}
    M_slow = M_slow * exp(-lambda_slow) + y_{t-1}
    """
    return HawkesPredictor.predict_dual_decay(params, series)


def global_loss(params: np.ndarray, datasets) -> float:
    if (params <= 0).any():
        return 1e12
    mu_fast, mu_slow, h_base, lam_fast, lam_slow = params
    if lam_fast <= lam_slow:
        return 1e12  # 违反双衰减约束，直接惩罚

    total = 0.0
    for ev in datasets:
        train_val = np.concatenate([ev["train"], ev["val"]])
        pred = rollout(params, train_val)
        total += compute_mse(train_val, pred)
    return total / len(datasets)


def evaluate_split(params: np.ndarray, datasets, split: str):
    """
    split: "train_val" 或 "test"
    返回 (avg_mse, avg_mape, per_dataset_list)
    """
    mu_fast, mu_slow, h_base, lam_fast, lam_slow = params
    if lam_fast <= lam_slow or (params <= 0).any():
        raise ValueError("参数不满足正值或 lambda_fast > lambda_slow 的约束。")
    per_ds = []
    total = 0.0
    total_mape = 0.0
    for ev in datasets:
        series = (
            np.concatenate([ev["train"], ev["val"]]) if split == "train_val" else ev["test"]
        )
        pred = rollout(params, series)
        mse = compute_mse(series, pred)
        mape = compute_mape(series, pred)
        per_ds.append((ev["name"], mse, mape))
        total += mse
        total_mape += mape
    n = len(datasets)
    return total / n, total_mape / n, per_ds


def main():
    datasets = load_all_datasets("datasets")
    print(f"共加载 {len(datasets)} 个事件，用于全局优化。")

    bounds = [
        (1e-6, 500.0),   # mu_fast
        (1e-6, 500.0),   # mu_slow
        (1e-6, 500.0),   # H_base
        (0.5, 5.0),      # lambda_fast
        (0.01, 2.0),     # lambda_slow
    ]

    init_guesses = [
        np.array([5.0, 2.0, 5.0, 3.5, 0.3]),
        np.array([3.0, 1.0, 10.0, 4.0, 0.5]),
        np.array([2.0, 2.0, 20.0, 2.5, 0.4]),
        np.array([1.0, 3.0, 30.0, 3.0, 0.2]),
        np.array([0.8, 0.8, 50.0, 2.0, 0.1]),
    ]

    best_params = None
    best_loss = float("inf")

    for idx, x0 in enumerate(init_guesses, start=1):
        print(f"\n[Start {idx}] init={x0.tolist()}")

        def wrapped(x):
            loss = global_loss(x, datasets)
            print(f"  params={x} loss={loss:.6f}")
            return loss

        res = minimize(
            wrapped,
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
        )
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_params = np.array(res.x, dtype=float)
            print(f"  -> New best loss {best_loss:.6f}")

    if best_params is None:
        raise RuntimeError("未找到可行解")

    print("\nGlobal Optimal Parameters (mu_fast, mu_slow, H_base, lambda_fast, lambda_slow):")
    print(best_params.tolist())
    print(f"Train+Val avg MSE (优化目标): {best_loss:.6f}")

    train_mse, train_mape, train_per = evaluate_split(best_params, datasets, split="train_val")
    print(f"Train+Val avg MAPE: {train_mape:.4f}%")

    test_mse, test_mape, per_ds = evaluate_split(best_params, datasets, split="test")
    print(f"Test avg MSE: {test_mse:.6f}")
    print(f"Test avg MAPE: {test_mape:.4f}%")
    print("Per-dataset test MSE / MAPE:")
    for name, l, mape in sorted(per_ds):
        print(f"  {name}: MSE={l:.6f}, MAPE={mape:.4f}%")


if __name__ == "__main__":
    main()
