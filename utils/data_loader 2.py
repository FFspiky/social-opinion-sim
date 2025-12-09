import glob
import os
from typing import List, Dict

import numpy as np
import pandas as pd


def load_all_datasets(
    data_dir: str = "datasets",
    random_test: bool = False,
    seed: int = 42,
    normalize: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """
    加载 data_dir 下的所有 CSV。
    默认按时间顺序 80/10/10 切分；
    若 random_test=True，则随机抽取 10% 时间点为 test（保持时间顺序输出），其余按时间顺序 80/10 切分。
    若 normalize=True，则按单个事件的 max 绝对值进行归一化，并返回 scale。
    """
    if not os.path.isdir(data_dir):
        print(f"[data_loader] 数据目录不存在：{data_dir}")
        return []

    file_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not file_paths:
        print(f"[data_loader] 数据目录下没有 CSV：{data_dir}")
        return []
    events = []
    rng = np.random.default_rng(seed)

    for path in file_paths:
        df = pd.read_csv(path)
        if "heat" not in df.columns:
            raise ValueError(f"{path} 缺少 heat 列")

        heat = df["heat"].astype(float).to_numpy()
        scale = 1.0
        if normalize:
            scale = float(np.max(np.abs(heat)))
            scale = max(scale, 1e-6)
            heat = heat / scale
        total = len(heat)

        if random_test:
            test_size = max(1, int(total * 0.1))
            test_idx = set(rng.choice(total, size=test_size, replace=False).tolist())
            remain_idx = [i for i in range(total) if i not in test_idx]
            test = heat[list(sorted(test_idx))]
            remain = heat[remain_idx]
            train_end = int(len(remain) * 0.8)
            val_end = train_end + int(len(remain) * 0.1)
            train = remain[:train_end]
            val = remain[train_end:val_end]
        else:
            train_end = int(total * 0.8)
            val_end = train_end + int(total * 0.1)
            train = heat[:train_end]
            val = heat[train_end:val_end]
            test = heat[val_end:]

        event = {
            "name": os.path.splitext(os.path.basename(path))[0],
            "train": train,
            "val": val,
            "test": test,
            "scale": scale,
        }
        events.append(event)

    print(f"已加载 {len(events)} 个事件数据")
    return events
