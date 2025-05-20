import os
import torch
import json
import random
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration

def setup_model():
    model_id = "/home/share/llava-hf/llava-1.5-7b-hf"

    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        device_map={"": "cuda:1"},
        torch_dtype=torch.float32
    )

    model = model.to("cuda:1")
    return processor, model

def load_instruct_dataset(processor, data, start=0, end=100):
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
                image_path = os.path.join(data, item["image"])
                if not os.path.exists(image_path):
                    print(f"Error: Image not found: {image_path}")
                    break  # Skip this example

                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    break

                con.append({
                    "role": role,
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": sentence["value"]}
                    ]
                })
            else:
                con.append({
                    "role": role,
                    "content": [{"type": "text", "text": sentence["value"]}]
                })
        if len(con) > 0:
            examples.append(con)
    return examples,image

@torch.no_grad()
def benchmark_instruct_tuning(model, processor, examples,image):
    total_tokens = 0
    total_time = 0.0
    speed_records = []

    for con in tqdm(examples, desc="Benchmarking Instruct Samples"):
        try:
            text = processor.apply_chat_template(con, tokenize=False, add_generation_prompt=True)

            # Extract image from first message

            inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)

            # Optional: clip input length if needed
            max_len = getattr(model.config, "max_position_embeddings", None)
            if max_len is not None and inputs["input_ids"].shape[1] > max_len:
                inputs["input_ids"] = inputs["input_ids"][:, -max_len:]

            # Use CUDA events for accurate timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=1,
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
    data = "/home/share/train2017"  # 指向 COCO 图像目录
    instruct_examples,image = load_instruct_dataset(processor, data, start=0, end=100)
    benchmark_instruct_tuning(model, processor, instruct_examples,image)
