import cv2
import math
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)
from datasets import load_dataset
import json
from PIL import Image


def sample_frames(video_path):
    frame_list = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    sample_times = [t for t in range(1, math.ceil(duration))]

    for t in sample_times:
        target_frame = int(t * fps)
        if target_frame >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        frame = np.transpose(frame, (1, 0, 2))

        pil_image = Image.fromarray(frame, mode="RGB")
        frame_list.append(pil_image)

    cap.release()
    return frame_list


def setup_model():
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    device = 'cuda:2'
    model_id = "%TARGET_PATH%"

    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to(device)
    return processor, model


def process_video(video_path, processor, model,i):
    model.eval()
    device = 'cuda:2'

    new_examples = {
        "input_ids": [],
        "pixel_values": [],
        "hidden_states":[],
        "input_emb":[],
        "prompt_len": []
    }
    output_subdir = '%DATA_PATH%'

    frames = sample_frames(video_path)
    frame_num = len(frames)
    prompt = "<image>\n" * frame_num +"Describe what happen in the video?"
    inputs = processor(images=frames, text=prompt, return_tensors="pt").to(device, torch.float16)
    new_examples["input_ids"]=inputs.input_ids.cpu()
    new_examples["pixel_values"]=inputs.pixel_values.cpu()
    outputs  = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=False,
    output_hidden_states=True,
    return_dict_in_generate=True
)
    generated_ids = outputs.sequences
    

    all_hidden_states = outputs.hidden_states


    layer0_hidden = []
    layer_last_hidden = []
    for step_hidden in all_hidden_states:
        layer0_hidden.append(step_hidden[0])
        layer_last_hidden.append(step_hidden[-1])

    layer0_stacks = torch.cat(layer0_hidden, dim=1)
    layer_last_stacks = torch.cat(layer_last_hidden, dim=1)

    seq_len =int(inputs.input_ids.shape[1])

    new_examples["hidden_states"]=layer_last_stacks.cpu()
    new_examples["input_emb"]=layer0_stacks.cpu()
    new_examples["prompt_len"]=seq_len

    output_path = os.path.join(output_subdir, f"{i}.pt")
    torch.save(new_examples, output_path)



video_dir = "%K400_PATH%"
processor, model = setup_model()
i = 0
for root, _, files in os.walk(video_dir):
    selected_files = sorted(files)[:20]
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