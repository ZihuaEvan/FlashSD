
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


def load_instruct_dataset(processor, start=0, end=100):
    ds = load_dataset("json", data_files="/home/wzy/eagle/EAGLE_EYE/eagle_eye/ge_data/llava_instruct_150k.json")["train"]
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(start, end))

    examples = []
    roles = {"human": "user", "gpt": "assistant"}

    for item in ds:
        source = item["conversations"]
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        con = []
        for i, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if role == "assistant":
                sentence["value"] += processor.tokenizer.eos_token
            if i == 0:
                con.append({
                    "role": role,
                    "content": [
                        {"type": "image", "image": os.path.join(data, item["image"])},
                        {"type": "text", "text": sentence["value"]}
                    ]
                })
            else:
                con.append({
                    "role": role,
                    "content": [{"type": "text", "text": sentence["value"]}]
                })
        examples.append(con)
    return examples

@torch.no_grad()
def benchmark_instruct_tuning(model, processor, examples):
    total_tokens = 0
    total_time = 0.0
    speed_records = []

    for con in tqdm(examples, desc="Benchmarking Instruct Samples"):
        try:
            text = processor.apply_chat_template(con, tokenize=False, add_generation_prompt=True)

            image_inputs, video_inputs, video_kwargs = process_vision_info(con, return_video_kwargs=True)
            inputs = processor(text=text, images=image_inputs, return_tensors="pt", **video_kwargs).to(model.device)

            # Use CUDA events for accurate timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=0,
                output_hidden_states=False,
                return_dict_in_generate=True
            )

            end_event.record()
            torch.cuda.synchronize()
            generation_time = start_event.elapsed_time(end_event) / 1000.0

            input_length = inputs["input_ids"].shape[1]
            output_length = outputs.sequences.shape[1]
            new_tokens = output_length - input_length

            if generation_time > 0:
                speed = new_tokens / generation_time
                speed_records.append(speed)
                total_tokens += new_tokens
                total_time += generation_time

            del inputs, outputs
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error: {e}")
            continue

    if len(speed_records) == 0:
        print("No valid samples processed.")
        return

    print("\n===== 指令调优生成速度报告 =====")
    print(f"样本数: {len(speed_records)}")
    print(f"总生成Token数: {total_tokens}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均速度: {total_tokens / total_time:.2f} tokens/s")
    print(f"中位数速度: {sorted(speed_records)[len(speed_records)//2]:.2f} tokens/s")

if __name__ == "__main__":
    processor, model = setup_model()
    data = "/home/share/train2017"  # 修改为你的图像根目录
    instruct_examples = load_instruct_dataset(processor, start=0, end=100)
    benchmark_instruct_tuning(model, processor, instruct_examples)