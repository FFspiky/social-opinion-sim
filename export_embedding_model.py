# export_embedding_model.py
from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"
save_path = "models/all-MiniLM-L6-v2"   # 将模型保存到项目内的 models 目录

print(f"加载模型: {model_name}")
model = SentenceTransformer(model_name)

print(f"保存模型到: {save_path}")
model.save(save_path)

print("完成。")
