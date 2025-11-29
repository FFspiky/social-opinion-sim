import glob
import os
from typing import List, Dict

import numpy as np
import pandas as pd


def load_all_datasets(data_dir: str = "datasets") -> List[Dict[str, np.ndarray]]:
    """
    加载 data_dir 下的所有 CSV，并按 80/10/10 时间顺序切分。
    返回每个事件的名称与 train/val/test 热度数组。
    """
    file_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    events = []

    for path in file_paths:
        df = pd.read_csv(path)
        if "heat" not in df.columns:
            raise ValueError(f"{path} 缺少 heat 列")

        heat = df["heat"].astype(float).to_numpy()
        total = len(heat)
        train_end = int(total * 0.8)
        val_end = train_end + int(total * 0.1)

        event = {
            "name": os.path.splitext(os.path.basename(path))[0],
            "train": heat[:train_end],
            "val": heat[train_end:val_end],
            "test": heat[val_end:],
        }
        events.append(event)

    print(f"已加载 {len(events)} 个事件数据")
    return events
