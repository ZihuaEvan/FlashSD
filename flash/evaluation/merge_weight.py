import os
import torch
from safetensors.torch import load_file
from transformers import AutoConfig
import sys
sys.path.append('/home/wzy/eagle/EAGLE_EYE/eagle_eye/model')
from cnets import Model
from configs import EConfig
save_dir = "/home/wzy/eagle/EAGLE_EYE/eagle_eye/weights/qwen2.5vl_video/state_18"

config = EConfig.from_pretrained('/home/wzy/eagle/EAGLE_EYE/eagle_eye/weights/qwen2.5vl_video/state_18/config.json')

model = Model(config, load_emb=True, path='/home/wzy/eagle/EAGLE_EYE/Qwen2.5-VL-7B-Instruct')

# 自动获取所有 model_X.safetensors 文件
safetensors_files = sorted(
    [f for f in os.listdir(save_dir) if f.startswith("model_") and f.endswith(".safetensors")]
)

# 加载所有分片的 state_dict
merged_state_dict = {}
for f in safetensors_files:
    shard = load_file(os.path.join(save_dir, f))
    merged_state_dict.update(shard)


# 加载合并的权重
model.load_state_dict(merged_state_dict)

# 保存为 pytorch_model.bin
torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

print("✅ 已合并并保存为 pytorch_model.bin")

