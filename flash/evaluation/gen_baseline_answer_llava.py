import argparse
import json
import os
import time

import shortuuid
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from flash.model.utils import prepare_logits_processor


# ---------------------------------------------------------------------------
# Baseline forward: pure autoregressive generation with KV cache.
# No draft head, no tree, no speculative decoding — just the base model.
# ---------------------------------------------------------------------------
@torch.inference_mode()
def baseline_forward(
    input_ids,
    pixel_values,
    model,
    tokenizer,
    logits_processor=None,
    max_new_tokens=512,
    max_length=4096,
):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    input_ids = input_ids.clone()
    input_len = input_ids.shape[1]
    new_token = 0

    # First forward: full prompt + image
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values

    for idx in range(max_new_tokens):
        if logits_processor is not None:
            logits = logits_processor(None, outputs.logits[:, -1, :])
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        new_token += 1

        # Stopping conditions (only inspect newly generated tail)
        if next_token.item() == tokenizer.eos_token_id:
            break
        if new_token >= max_new_tokens:
            break
        if input_ids.shape[1] >= max_length:
            break

        # Subsequent forward: single token + KV cache
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

    return input_ids, new_token, idx


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_eval(
    base_model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    datapath,
    max_new_token,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    temperature,
):
    with open(os.path.expanduser(question_file), "r") as f:
        js = json.load(f)

    images = js.get("images", [])
    data = images[question_begin:question_end]

    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(data) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(data), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                model_id,
                data[i: i + chunk_size],
                answer_file,
                datapath,
                max_new_token,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    base_model_path,
    model_id,
    questions,
    answer_file,
    datapath,
    max_new_token,
    num_gpus_per_model,
    max_gpu_memory,
    temperature,
):
    # ----- Load base VLM directly (no draft head) -----
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
        attn_implementation="eager",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model_path)
    tokenizer = processor.tokenizer

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    print("warmup ...")
    for j in range(min(3, len(questions))):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Provide a detailed description of the given image."},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image = Image.open(os.path.join(datapath, questions[j]["file_name"]))
        inputs = processor(images=image, text=text, return_tensors="pt")
        input_ids = inputs.input_ids
        pixel_values = inputs.pixel_values

        torch.cuda.synchronize()
        _ = baseline_forward(
            input_ids=torch.as_tensor(input_ids).cuda(),
            pixel_values=torch.as_tensor(pixel_values).cuda(),
            model=model,
            tokenizer=tokenizer,
            logits_processor=logits_processor,
            max_new_tokens=max_new_token,
        )
        torch.cuda.synchronize()
    print("Warmup done")

    torch.manual_seed(123)
    for question in tqdm(questions):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Provide a detailed description of the given image."},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image = Image.open(os.path.join(datapath, question["file_name"]))
        inputs = processor(images=image, text=text, return_tensors="pt")
        input_ids = inputs.input_ids
        pixel_values = inputs.pixel_values

        torch.cuda.synchronize()
        start_time = time.time()
        output_ids, new_token, idx = baseline_forward(
            input_ids=torch.as_tensor(input_ids).cuda(),
            pixel_values=torch.as_tensor(pixel_values).cuda(),
            model=model,
            tokenizer=tokenizer,
            logits_processor=logits_processor,
            max_new_tokens=max_new_token,
        )
        torch.cuda.synchronize()
        total_time = time.time() - start_time

        output_ids = output_ids[0][len(input_ids[0]):]
        new_token = output_ids.shape[-1]

        output = tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        if tokenizer.eos_token and output.find(tokenizer.eos_token) > 0:
            output = output[: output.find(tokenizer.eos_token)]
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()

        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "response": output,
                "idx": idx,
                "new_tokens": new_token,
                "wall_time": total_time,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, default="path/to/llava-1.5-7b-hf",
                        help="1")
    parser.add_argument(
        "--load-in-8bit",
        action="store_false",
        help="Use 8-bit quantization",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="llava-v1.5-7b-hf-fp16-baseline",
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="COCO-caption",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--question-end",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--answer-dir",
        type=str,
        default="./outputs",
    )
    parser.add_argument("--datajson", type=str, default="path/to/coco/annotations/captions_val2014.json",help="Path to the input JSON file containing questions or data.")  
    parser.add_argument("--datapath", type=str, default="path/to/coco/val2014",help="Name or path of the dataset to be used for evaluation.")  

    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num-gpus-total",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray
        ray.init()

    answer_file = os.path.join(
        args.answer_dir, args.bench_name, f"{args.model_id}.jsonl"
    )
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    print(f"Output to {answer_file}")

    run_eval(
        base_model_path=args.base_model_path,
        model_id=args.model_id,
        question_file=args.datajson,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        datapath=args.datapath,
        max_new_token=args.max_new_token,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        temperature=args.temperature,
    )
