import os
import torch
import time
from tqdm import tqdm
import random
from transformers import AutoProcessor, LlavaForConditionalGeneration

# --------------------------------------
# 工具函数
# --------------------------------------
def setup_model():
    """初始化模型和处理器"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    device = 'cuda:0'
    model_id = "/home/share/llava-hf/llava-1.5-7b-hf"

    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": device}
    )
    return processor, model

def load_pt_files(directory):
    """加载目录下所有.pt文件路径（按字母顺序排序）"""
    pt_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pt'):
                pt_files.append(os.path.join(root, file))
    pt_files.sort()  # 确保文件列表有序
    return pt_files

# --------------------------------------
# 核心统计逻辑（添加采样逻辑）
# --------------------------------------
def benchmark_generation(pt_dir, model, seed=42):
    """从.pt文件加载数据并统计生成速度（采样20%）"""
    device = model.device
    pt_files = load_pt_files(pt_dir)
    
    # 设置随机种子
    random.seed(seed)
    
    # 计算采样数量（至少1个）
    sample_size = max(1, int(len(pt_files) * 0.2))
    selected_files = random.sample(pt_files, sample_size)
    
    # 打印采样信息
    print(f"总文件数: {len(pt_files)}")
    print(f"采样数量: {sample_size} ({sample_size/len(pt_files):.1%})")
    print(f"固定随机种子: {seed}")

    total_tokens = 0
    total_time = 0.0
    speed_records = []
    
    model.eval()
    
    for pt_file in tqdm(selected_files, desc="Benchmarking"):
        try:
            # 加载数据
            data = torch.load(pt_file)
            input_ids = data["input_ids"].to(device)
            pixel_values = data["pixel_values"].to(device, dtype=torch.float16)
            
            # 构建输入字典
            inputs = {
                "input_ids": input_ids,
                "pixel_values": pixel_values
            }
            
            # 精确计时（使用CUDA事件）
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_event.record()
            
            # 执行生成
            with torch.no_grad():
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
            
            # 计算耗时（毫秒转秒）
            generation_time = start_event.elapsed_time(end_event) / 1000.0
            
            # 统计token数量
            input_length = inputs["input_ids"].shape[1]
            output_length = outputs.sequences.shape[1]
            new_tokens = output_length - input_length
            
            # 记录结果
            if generation_time > 0:
                tokens_per_sec = new_tokens / generation_time
                speed_records.append(tokens_per_sec)
                total_tokens += new_tokens
                total_time += generation_time
            
            # 清理显存
            del inputs, outputs
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError processing {os.path.basename(pt_file)}: {str(e)}")
            continue
    
    # 生成统计报告
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

# --------------------------------------
# 主函数
# --------------------------------------
if __name__ == "__main__":
    # 初始化模型
    processor, model = setup_model()
    
    # 指定.pt文件目录
    pt_dir = "/home/wzy/eagle/EAGLE_EYE/eagle_eye/ge_data/tmp/k400"  # 修改为你的实际路径
    
    fixed_seed = 42  # 可修改为任意整数
    
    # 执行统计
    benchmark_generation(pt_dir, model, seed=fixed_seed)