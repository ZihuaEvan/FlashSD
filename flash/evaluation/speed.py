import json
from transformers import AutoTokenizer
import numpy as np

# jsonl_file = "/home/dhz/eagle-eye/EAGLE_EYE/eagle_eye/outputs/COCO-caption/llava-v1.5-7b-hf-fp16-ee-temperature-1.0.jsonl"
# jsonl_file_base = "/home/dhz/eagle-eye/EAGLE_EYE/eagle_eye/outputs/COCO-caption/llava-v1.5-7b-hf-fp16-baseline-temperature-1.0.jsonl"

jsonl_file = "/home/dhz/eagle-eye/EAGLE_EYE/eagle_eye/outputs/COCO-caption/Qwen2.5-VL-7B-Instruct-fp16-ee-temperature-1.0.jsonl"
jsonl_file_base = "/home/dhz/eagle-eye/EAGLE_EYE/eagle_eye/outputs/COCO-caption/Qwen2.5-VL-7B-Instruct-fp16-baseline-temperature-1.0.jsonl"

data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)


total_time=0
total_token=0
total_idxs=0
speeds=[]
for datapoint in data:
    qid=datapoint['question_id']
    answer=datapoint['response']
    idx=datapoint['idx']
    tokens=datapoint['new_tokens']
    times = datapoint['wall_time']
    
    speeds.append(tokens/times)

    total_time+=times
    total_token+=tokens
    total_idxs+=idx

print(total_token)
print(total_idxs)

data = []
with open(jsonl_file_base, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)
        
total_time=0
total_token=0

speeds0=[]
for datapoint in data:
    answer=datapoint['response']
    tokens=datapoint['new_tokens']
    times = datapoint['wall_time']

    speeds0.append(tokens / times)
    total_time+=times
    total_token+=tokens
    


print('speed',np.array(speeds).mean())
print('speed0',np.array(speeds0).mean())
print("ratio",np.array(speeds).mean()/np.array(speeds0).mean())


