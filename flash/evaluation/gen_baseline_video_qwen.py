import os
import torch
import time
from tqdm import tqdm
import random
from transformers import AutoProcessor, LlavaForConditionalGeneration


def setup_model():
    model_id = "path/to/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map={"": "cuda:3"},
        torch_dtype=torch.float32
    )

    model = model.to("cuda:3")
    return processor, model

def load_pt_files(directory):

    pt_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pt'):
                pt_files.append(os.path.join(root, file))
    pt_files.sort()
    return pt_files


def benchmark_generation(pt_dir, model, seed=42):
    device = model.device
    pt_files = load_pt_files(pt_dir)
    

    random.seed(seed)
    
    sample_size = max(1, int(len(pt_files) * 0.2))
    

    total_tokens = 0
    total_time = 0.0
    speed_records = []
    
    model.eval()
    selected_files = pt_files[:sample_size]
    for i, file in enumerate(tqdm(selected_files, desc="Processing videos")):
        video_path = os.path.join(root, file)
        try:
            con = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video":video_path,
                    "fps": 1.0,
                },
                {"type": "text", "text": "Describe what happen in the video?"},
            ],
        }
    ]
            text = processor.apply_chat_template(con, tokenize=False, add_generation_prompt=True)

            image_inputs, video_inputs, video_kwargs = process_vision_info(con, return_video_kwargs=True)
            inputs = processor(text=text, videos=video_inputs,return_tensors="pt",**video_kwargs).to(model.device, torch.float16)
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_event.record()
            

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=0.0,
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
            print(f"\nError processing {os.path.basename(pt_file)}: {str(e)}")
            continue

    if len(speed_records) == 0:
        print("No valid samples processed.")
        return
    
    avg_speed = total_tokens / total_time

    
    print("\n===== 生成速度统计报告 =====")
    print(f"已处理样本数: {len(speed_records)}")
    print(f"总生成Token数: {total_tokens}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均速度: {avg_speed:.2f} tokens/s")
    print(f"中位数速度: {sorted(speed_records)[len(speed_records)//2]:.2f} tokens/s")


if __name__ == "__main__":

    processor, model = setup_model()

    pt_dir = "path/to/k400_features"
    
    fixed_seed = 42
    
    # 执行统计
    benchmark_generation(pt_dir, model, seed=fixed_seed)
