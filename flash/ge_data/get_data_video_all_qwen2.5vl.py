
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

from qwen_vl_utils import process_vision_info


def setup_model():
    model_id = "%TARGET_PATH%"
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map={"": "cuda:2"},  
        torch_dtype=torch.float32
    )
    
    model = model.to("cuda:2")
    return processor, model



def process_video(video_path, processor, model,i):
    output_subdir = '%DATA_PATH%'
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    new_examples = {
       "input_ids": [],
        "inputs_embeds": [],
        "hidden_states": [],
        "loss_mask": [],
    }
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
    inputs = processor(text=text, videos=video_inputs,return_tensors="pt",**video_kwargs).to(model.device, torch.float32)

    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            output_hidden_states=True,  
            return_dict_in_generate=True  
        )
        generated_ids = outputs.sequences[:,:-1]

        layer0_hidden = []
        layer_last_hidden = []
        
        for step_hidden in outputs.hidden_states:
            layer0_hidden.append(step_hidden[0])        # 第0层
            layer_last_hidden.append(step_hidden[-1])   # 最后一层

        inputs_embeds = torch.cat(layer0_hidden, dim=1)  # 形状: (1, total_seq_len, hidden_size)
        hidden_states = torch.cat(layer_last_hidden, dim=1)


        loss_mask = torch.ones_like(generated_ids)

        loss_mask[:,:inputs.input_ids.shape[-1]] = 0

        new_examples["input_ids"] = generated_ids.cpu()[0]
        new_examples["inputs_embeds"] = inputs_embeds.cpu()[0]
        new_examples["hidden_states"] = hidden_states.cpu()[0]
        new_examples["loss_mask"] = loss_mask.cpu()[0]

    
    current_length = len(os.listdir(output_subdir))
    idx = current_length
    torch.save(new_examples, f"{output_subdir}/{idx}.pt")




video_dir = "%K400_PATH%"
processor, model = setup_model()
i = 0
for root, _, files in os.walk(video_dir):
    # 对文件进行排序并取前20个
    selected_files = sorted(files)[:20]
    # 使用enumerate获取序号i，从0开始
    print(root)
    for i, file in enumerate(tqdm(selected_files, desc="Processing videos")):
        video_path = os.path.join(root, file)
        if os.path.splitext(file)[1].lower() in ['.mp4','.MP4']:
            try:
                process_video(
                    video_path,
                    processor,
                    model,
                    i
                )
                i=i+1
            except Exception as e:
                print(f"Error processing {video_path}: {e}")