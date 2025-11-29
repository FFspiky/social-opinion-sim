import os
import numpy as np
import pandas as pd
import random

# 配置
DATASET_DIR = "datasets"
TIME_STEPS = 350  # 总时间步长，足够切分 (Train~280, Val~35, Test~35)

# 确保目录存在
os.makedirs(DATASET_DIR, exist_ok=True)

# 10大类典型事件列表 (Name, Type, Intensity_Base, Pattern_Type)
# Pattern_Type: 
#   'burst': 突发型 (如地震、佩洛西窜台) - 瞬间达峰，快速衰减
#   'wave': 波动型 (如两会、经济复苏) - 缓慢爬坡，维持较久
#   'meme': 病毒型 (如淄博烧烤、梗) - 多波次爆发，长尾
events = [
    # 1. 政治引领
    ("01_Politics_20th_Congress", "wave", 800),
    ("02_Politics_Belt_Road", "wave", 600),
    ("03_Politics_Silver_Eco", "wave", 400),
    # 2. 国际关系
    ("04_Intl_Pelosi_Visit", "burst", 1000), # 极高热度
    ("05_Intl_Saudi_Iran", "burst", 700),
    ("06_Intl_Balloon_Incident", "wave", 500),
    ("07_Intl_Israel_Palestine", "wave", 600),
    # 3. 经济复苏
    ("08_Eco_Zibo_BBQ", "meme", 900), # 现象级
    ("09_Eco_Huawei_Mate60", "burst", 850),
    ("10_Eco_Real_Estate", "wave", 500),
    ("11_Eco_Canton_Fair", "wave", 300),
    # 4. 社会民生
    ("12_Social_Covid_Shift", "burst", 950),
    ("13_Social_Retirement_Delay", "wave", 600),
    ("14_Social_Healthcare", "wave", 400),
    # 5. 法治建设
    ("15_Law_Tangshan_Case", "burst", 900),
    ("16_Law_Corruption_Li", "burst", 500),
    ("17_Law_Article20_Movie", "meme", 600),
    # 6. 科技创新
    ("18_Tech_Sora_AI", "burst", 800),
    ("19_Tech_Change6_Moon", "burst", 700),
    ("20_Tech_C919_Flight", "burst", 600),
    # 7. 突发灾害
    ("21_Disaster_Jishishan_Quake", "burst", 850),
    ("22_Disaster_South_Floods", "wave", 500),
    ("23_Disaster_Tornado", "burst", 600),
    # 8. 教育青年
    ("24_Edu_Grad_Exam", "wave", 500),
    ("25_Edu_KongYiji", "meme", 700),
    ("26_Edu_Banwei_Work", "meme", 400),
    # 9. 文体娱乐
    ("27_Ent_Olympics_Diving", "burst", 900),
    ("28_Ent_Tax_Scandal", "burst", 800),
    ("29_Ent_Concert_Economy", "wave", 600),
    ("30_Ent_Esports_Win", "burst", 750),
    # 10. 网络亚文化
    ("31_Sub_MaBaoguo_Meme", "meme", 500),
    ("32_Sub_Meme_Evolution", "meme", 300),
    ("33_Sub_MBTI_Trends", "wave", 200),
    ("34_Sub_Army_Coat", "meme", 400),
]

def generate_series(pattern, base_intensity):
    """生成模拟真实舆情的合成数据"""
    time = np.arange(TIME_STEPS)
    heat = np.zeros(TIME_STEPS)
    
    # 基础噪音
    noise = np.random.normal(0, 10, TIME_STEPS)
    heat += np.abs(noise) + 50 # 基础底噪
    
    # 根据模式生成主峰和余波
    if pattern == 'burst':
        # 突发型：陡峭上升，指数衰减
        peak_time = random.randint(30, 80) # 发生在早期
        decay = random.uniform(0.05, 0.1)
        # 主峰
        heat[peak_time:] += base_intensity * np.exp(-decay * (time[peak_time:] - peak_time))
        # 余波 (Aftershocks)
        for _ in range(2):
            sub_peak = random.randint(peak_time + 20, TIME_STEPS - 50)
            sub_intensity = base_intensity * random.uniform(0.1, 0.3)
            heat[sub_peak:] += sub_intensity * np.exp(-decay * 1.5 * (time[sub_peak:] - sub_peak))
            
    elif pattern == 'wave':
        # 波动型：上升缓慢，持续平台期，缓慢下降
        start_time = random.randint(20, 50)
        duration = random.randint(50, 100)
        
        # 使用高斯函数模拟
        center = start_time + duration / 2
        sigma = duration / 4
        heat += base_intensity * np.exp(-0.5 * ((time - center) / sigma) ** 2)
        
        # 叠加小的波动
        heat += base_intensity * 0.1 * np.sin(time / 10)
        
    elif pattern == 'meme':
        # 病毒型：多波次，随机性强
        num_waves = random.randint(3, 5)
        for _ in range(num_waves):
            wave_time = random.randint(20, TIME_STEPS - 50)
            wave_intensity = base_intensity * random.uniform(0.4, 1.0)
            decay = random.uniform(0.03, 0.08)
            heat[wave_time:] += wave_intensity * np.exp(-decay * (time[wave_time:] - wave_time))
            
    # 确保没有负数
    heat = np.clip(heat, 0, None)
    return pd.DataFrame({'time': time, 'heat': heat})

# 执行生成
print(f"开始生成 {len(events)} 个典型舆情事件数据...")
for name, pattern, intensity in events:
    # 增加随机性，使每个事件即便类型相同也略有不同
    adjusted_intensity = intensity * random.uniform(0.8, 1.2)
    df = generate_series(pattern, adjusted_intensity)
    
    filename = f"{DATASET_DIR}/{name}.csv"
    df.to_csv(filename, index=False)
    print(f"  [生成] {filename} (模式: {pattern})")

print(f"\n✅ 全部完成！数据已保存在 '{DATASET_DIR}/' 目录。")
print("现在你可以运行你的 train.py 来读取这些文件了。")