import argparse
import copy
import re
parser = argparse.ArgumentParser(description="sp")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=50000)
parser.add_argument("--index", type=int, default=1)
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])
parser.add_argument(
    "--outdir", type=str, default="%DATA_PATH%"
)
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
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



bigname = "%TARGET_PATH%"

data = "%COCO2017_PATH%"


def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break

    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length


def build_dataset_rank(
    processor,
    split="train",
    select=None,
):
    ds = load_dataset(
        "json", data_files="./llava_instruct_150k.json"
    )
    ds = ds["train"]
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    #ds1 = ds.select(range(args.start, args.end))
    # ds1 = ds.select(range(100,200))
    # dst=ds.select(range(200,300))
    # ds2=ds.select(range(300,len(ds)))
    original_columns1 = ds1.column_names
    # original_columns2 = ds2.column_names
    num_proc = 1
    tokenizer = processor.tokenizer
    seen_ids = set()
    def preprocess_function_with_seen_ids(examples):
        return preprocess_function(examples, seen_ids)
    def preprocess_function(examples, seen_ids):
        new_examples = {
            "conversation": [],
            "input_ids": [],
            "pixel_values": [],
            "loss_mask": [],
        }
        for i in range(len(examples["id"])):
            if examples["id"][i] in seen_ids:
                continue  
            seen_ids.add(examples["id"][i])  
            url = data + "/" + examples["image"][i]
            roles = {"human": "user", "gpt": "assistant"}
            source = examples["conversations"][i]
            con = []

            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]

            for i,sentence in enumerate(source):
                role = roles[sentence["from"]]
                if role == "assistant":
                    sentence["value"] += tokenizer.eos_token
                if i==0 :
                    con.append({
                        "role": role,
                        "content": [
                            {"type": "image","image": url},
                            {"type": "text", "text": sentence["value"]}
                        ]
                    })
                else :
                    con.append({
                        "role": role,
                        "content": [
                            {"type": "text", "text": sentence["value"]}
                        ]
                    })


            text = con[0]['content'][1]['text']
            cleaned_text = re.sub(r'\n?<image>\n?', '', text)
            con[0]['content'][1]['text'] = cleaned_text

            conversation= processor.apply_chat_template(con, tokenize=False, add_generation_prompt=False)
            
            image = Image.open(url)
            
            inputs = processor(images=image, text=conversation, return_tensors='pt')

            input_ids=inputs['input_ids'][0]
            pixel_values=inputs['pixel_values']

            loss_mask = torch.ones_like(input_ids)

            sep = " " + "ASSISTANT" + ": "

            total_len = int(input_ids.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(tokenizer.eos_token)
            cur_len = 1

            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == " ":
                    break
                turn_len = len(processor(images=image, text=turn, return_tensors='pt').input_ids[0])

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(processor(images=image, text=parts[0], return_tensors='pt').input_ids[0]) - 2

                # if i != 0 and not tokenizer.legacy:
                #     # The legacy and non-legacy modes handle special tokens differently
                #     instruction_len -= 1

                # Ignore the user instructions
                loss_mask[cur_len : cur_len + instruction_len] = 0
                cur_len += turn_len

                # if i != 0 and not tokenizer.legacy:
                #     # The legacy and non-legacy modes handle special tokens differently
                #     cur_len -= 1

            loss_mask[cur_len:] = 0

            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["pixel_values"].append(pixel_values)
            new_examples["loss_mask"].append(loss_mask[None, :])

        return new_examples

    ds1 = ds1.map(
        preprocess_function_with_seen_ids,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False,
    )

    # ds1 = ds1.filter(lambda x: len(x["input_ids"]) < 1024, batched=False)
    # ds1 = ds1.filter(lambda x: x['queryf'] not in gqs, batched=False)
    # ds1 = ds1.filter(lambda x: "Are there any tips in regards to teaching" in x['queryf'], batched=False)

    ds1.set_format(type="torch")
    # ds2.set_format(type="torch")
    # dst.set_format(type="torch")
    return ds1

bigmodel = LlavaForConditionalGeneration.from_pretrained(
    bigname, device_map="auto", torch_dtype=torch.float16,attn_implementation='eager')


bigprocessor = AutoProcessor.from_pretrained(bigname, use_fast=False)
bigprocessor.patch_size =bigmodel.config.vision_config.patch_size
ds = build_dataset_rank(bigprocessor)
print(ds)

bigmodel.eval()


@torch.no_grad()
def ge(data):
    input_ids = data["input_ids"]
    pixel_values = data["pixel_values"]
    attention_mask = torch.ones_like(input_ids)

    outs_big = bigmodel(
        input_ids.cuda(),
        pixel_values.cuda(),
        attention_mask.cuda(),
        output_hidden_states=True,
    )
    inputs_embeds = outs_big.hidden_states[0]
    hidden_state_big = outs_big.hidden_states[-1]

    # max_prob_tokens_big = torch.argmax(outs_big.logits, dim=-1)
    # probs = torch.softmax(outs_big.logits, dim=-1)
    # maxp = probs[0].max(dim=1).values
    td = {
        "input_ids": input_ids.cpu()[0],
        "inputs_embeds": inputs_embeds.cpu()[0],
        "hidden_state": hidden_state_big.cpu()[0],
        "loss_mask": data["loss_mask"].cpu()[0],
    }
    return td


outdir = f"{args.outdir}/{args.index}"
if not os.path.exists(outdir):
    os.makedirs(outdir)


def writedata(name, data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length = len(os.listdir(name))
    idx = current_length
    torch.save(data_point, f"{name}/data_{idx}.ckpt")


for data in ds:
    outdata = ge(data)
    writedata(outdir, outdata)
