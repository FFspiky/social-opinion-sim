import math
from typing import Iterable, Optional

import numpy as np

__all__ = ["HawkesPredictor", "HawkesProcess"]


class HawkesProcess:
    """
    兼容旧接口的简单霍克斯过程占位实现，仅用于 simulate.py 导入不报错。
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, mu: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.events = []

    def simulate_event(self, current_time: float) -> bool:
        intensity = self.mu + sum(self.alpha * math.exp(-self.beta * (current_time - t)) for t in self.events)
        if np.random.random() < intensity:
            self.events.append(current_time)
            return True
        return False

    def get_event_count(self) -> int:
        return len(self.events)


class HawkesPredictor:
    """
    双衰减核（Dual-Decay Kernel）自激模型：
        pred_t = H_base + mu_fast * M_fast + mu_slow * M_slow
        M_fast = M_fast * exp(-lambda_fast) + y_{t-1}
        M_slow = M_slow * exp(-lambda_slow) + y_{t-1}

    约束：lambda_fast > lambda_slow 且参数均为正。
    """

    @staticmethod
    def predict_dual_decay(
        params: Iterable[float],
        series: Iterable[float],
    ) -> np.ndarray:
        """
        对给定真实序列 series 进行 teacher-forcing 滚动预测。
        params: [mu_fast, mu_slow, H_base, lambda_fast, lambda_slow]
        返回与 series 等长的预测数组。
        """
        mu_fast, mu_slow, h_base, lam_fast, lam_slow = [float(x) for x in params]
        if mu_fast <= 0 or mu_slow <= 0 or h_base <= 0 or lam_fast <= 0 or lam_slow <= 0:
            raise ValueError("All parameters must be positive.")
        if lam_fast <= lam_slow:
            raise ValueError("lambda_fast must be greater than lambda_slow.")

        decay_fast = math.exp(-lam_fast)
        decay_slow = math.exp(-lam_slow)

        series_arr = np.asarray(series, dtype=float)
        preds = np.zeros_like(series_arr, dtype=float)
        mem_fast = 0.0
        mem_slow = 0.0

        for i, y in enumerate(series_arr):
            preds[i] = h_base + mu_fast * mem_fast + mu_slow * mem_slow
            mem_fast = mem_fast * decay_fast + y
            mem_slow = mem_slow * decay_slow + y

        return preds

    # 为兼容性保留旧方法名，内部调用双衰减版本。
    @staticmethod
    def predict(
        params: Iterable[float],
        history_len: int,
        forecast_len: int,
        history: Optional[Iterable[float]] = None,
    ) -> np.ndarray:
        """
        兼容旧接口：history_len+forecast_len；仅当 forecast_len=0 时使用真实历史。
        """
        if forecast_len != 0:
            raise ValueError("Dual-decay predictor当前仅支持 forecast_len=0 的 teacher-forcing 预测。")

        if history is None:
            raise ValueError("需要提供 history 序列。")

        hist_arr = np.asarray(history, dtype=float)
        if len(hist_arr) > history_len:
            hist_arr = hist_arr[:history_len]
        return HawkesPredictor.predict_dual_decay(params, hist_arr)
