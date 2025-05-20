
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from datasets import load_dataset
import json
from PIL import Image
import os
import torch
import time
from tqdm import tqdm
import random

from qwen_vl_utils import process_vision_info


def setup_model():
    model_id = "/home/wzy/eagle/EAGLE_EYE/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map={"": "cuda:1"},  
        torch_dtype=torch.float32
    )
    
    model = model.to("cuda:1")
    return processor, model


def collect_video_paths(root_dir):
    video_paths = []
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):  # 可按需扩展格式
                video_paths.append(os.path.join(class_dir, file_name))
    return video_paths


# --------------------------------------
# 核心统计逻辑（添加采样逻辑）
# --------------------------------------
def benchmark_generation(root_dir, model, processor, seed=42, sample_size=100):
    """从K400 dataset中采样100个视频进行测速"""
    device = model.device
    all_video_paths = collect_video_paths(root_dir)

    print(f"视频总数: {len(all_video_paths)}")
    if len(all_video_paths) == 0:
        print("未找到任何视频文件")
        return

    # 设置随机种子并采样
    random.seed(seed)
    selected_videos = random.sample(all_video_paths, min(sample_size, len(all_video_paths)))
    print(f"采样视频数: {len(selected_videos)} (Seed: {seed})")

    total_tokens = 0
    total_time = 0.0
    speed_records = []

    model.eval()

    for video_path in tqdm(selected_videos, desc="Benchmarking"):
        try:
            con = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "fps": 1.0,
                        },
                        {"type": "text", "text": "Describe what happen in the video?"},
                    ],
                }
            ]

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_event.record()

            with torch.no_grad():
                text = processor.apply_chat_template(con, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs, video_kwargs = process_vision_info(con, return_video_kwargs=True)
                inputs = processor(text=text, videos=video_inputs, return_tensors="pt", **video_kwargs).to(device, torch.float32)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=1.0,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )

            end_event.record()
            torch.cuda.synchronize()

            generation_time = start_event.elapsed_time(end_event) / 1000.0
            input_length = inputs["input_ids"].shape[1]
            output_length = outputs.sequences.shape[1]
            new_tokens = output_length - input_length

            if generation_time > 0:
                tokens_per_sec = new_tokens / generation_time
                speed_records.append(tokens_per_sec)
                total_tokens += new_tokens
                total_time += generation_time

            del inputs, outputs
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n处理视频失败 {os.path.basename(video_path)}: {str(e)}")
            continue

    if len(speed_records) == 0:
        print("无有效样本被处理。")
        return

    avg_speed = total_tokens / total_time
    print("\n===== 生成速度统计报告 =====")
    print(f"已处理样本数: {len(speed_records)}")
    print(f"总生成Token数: {total_tokens}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均速度: {avg_speed:.2f} tokens/s")
    print(f"中位数速度: {sorted(speed_records)[len(speed_records) // 2]:.2f} tokens/s")
# --------------------------------------
# 主函数
# --------------------------------------
if __name__ == "__main__":
    # 初始化模型
    processor, model = setup_model()

    # 指定.pt文件目录
    root_dir = "/home/share/k400/val_256"  # 修改为你的实际路径

    fixed_seed = 42  # 可修改为任意整数

    # 执行统计
    benchmark_generation(root_dir, model, processor, seed=fixed_seed)