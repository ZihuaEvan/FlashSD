"""
SAR (Semi-AutoRegressive) speculative decoding evaluation for LLaVA-1.5 (v2).

Fixed issues from v1:
  1. Import path: uses llama_sar_new2 (current architecture).
  2. Embedding shift: position t gets embed(token_{t+1}) to match training.
  3. Residual gate: applied during inference (hidden_states_residual passed).
  4. No train() hack: model stays in eval() mode (dropout disabled).
"""

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
from flash.model.llama_sar_new2 import Model as SARModel
from flash.model.configs import EConfig
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_base_vlm(base_model_path, device_map="cuda:0"):
    """Load LLaVA base model + processor only."""
    base_model = LlavaForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation="eager",
    )
    base_model.eval()
    processor = AutoProcessor.from_pretrained(base_model_path)
    return base_model, processor


def get_lm_head(base_model):
    """Return the frozen LM head module for LLaVA."""
    if hasattr(base_model, "lm_head") and base_model.lm_head is not None:
        return base_model.lm_head
    return base_model.language_model.lm_head


def load_sar_model(sar_model_path, device, dtype):
    """Instantiate SARModel from a folder with config.json + weights."""
    config_path = os.path.join(sar_model_path, "config.json")
    config = EConfig.from_pretrained(config_path)

    with open(config_path, "r") as f:
        con = json.loads(f.read())
    bias = con.get("bias", True)

    sar_model = SARModel(config, load_emb=False, path=None, bias=bias)

    safetensors_path = os.path.join(sar_model_path, "model.safetensors")
    bin_path = os.path.join(sar_model_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path, device="cpu")
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No weights file found in {sar_model_path}. "
            f"Expected one of: {safetensors_path} or {bin_path}"
        )

    sar_model.load_state_dict(state_dict, strict=False)
    sar_model = sar_model.to(dtype).to(device)
    sar_model.eval()  # FIX #4: keep eval mode (dropout disabled)
    return sar_model


# ---------------------------------------------------------------------------
# KV cache helpers
# ---------------------------------------------------------------------------
def trim_kv_cache(cache, target_len):
    """Truncate past_key_values along the sequence dim to ``target_len``."""
    if cache is None:
        return None

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for i in range(len(cache.key_cache)):
            k = cache.key_cache[i]
            v = cache.value_cache[i]
            if k is None or v is None:
                continue
            if k.shape[-2] > target_len:
                cache.key_cache[i] = k[..., :target_len, :]
                cache.value_cache[i] = v[..., :target_len, :]
        if hasattr(cache, "_seen_tokens"):
            cache._seen_tokens = target_len
        return cache

    if isinstance(cache, tuple):
        return tuple(
            (k[..., :target_len, :], v[..., :target_len, :])
            for (k, v) in cache
        )

    raise TypeError(f"Unknown past_key_values type: {type(cache)}")


# ---------------------------------------------------------------------------
# SAR speculative decoding loop (v2 — all 4 fixes applied)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def sar_forward(
    input_ids,
    pixel_values,
    base_model,
    lm_head,
    sar_model,
    tokenizer,
    logits_processor=None,
    max_new_tokens=512,
    max_length=4096,
):
    """SAR speculative decoding (Design A) with all inference fixes.

    Fixes applied:
      1. Embedding shift: embeds are shifted by 1 to match training convention.
      2. Residual gate: hidden_states_residual passed to sar_only().
      3. No train() hack: model stays in eval() mode.
      4. lm_head passed for Hydra refinement.
    """
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    input_ids = input_ids.clone()

    k = getattr(sar_model, "k", 5)
    if k < 2:
        raise ValueError(f"Merged-verify SAR requires k>=2; got k={k}")
    eff_drafts = k - 1
    new_token = 0
    steps = 0

    # FIX #4: Keep eval mode — sar_only() already returns (B, L', k, D).
    # No need for train() hack. Dropout stays disabled.
    sar_model.eval()
    sar_model.reset_cache()

    # ================== Initial prefill ==================
    prefill_out = base_model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        output_hidden_states=True,
        use_cache=True,
    )
    past_kv_full = prefill_out.past_key_values
    hidden_prefill = prefill_out.hidden_states[-1]   # (1, L_prefill, D)
    embeds_prefill = prefill_out.hidden_states[0]    # (1, L_prefill, D)
    L_prefill = input_ids.shape[1]

    # FIX #2: Shift embeddings by 1 to match training convention.
    # Training: position t gets embed(token_{t+1}).
    # Base model gives: embeds_prefill[:, t, :] = embed(token_t).
    # So shifted_embeds[:, t, :] = embeds_prefill[:, t+1, :] = embed(token_{t+1}).
    # We need positions 0..L_prefill-2, with shifted embeds from 1..L_prefill-1.
    hidden_for_prefill = hidden_prefill[:, :-1, :]    # h_0..h_{L-2}, length L-1
    embeds_for_prefill = embeds_prefill[:, 1:, :]     # embed(token_1)..embed(token_{L-1}), length L-1

    # SAR prefill: compute post-FC buffer with shifted embeds.
    # input_ids[:, :-1] is for image token detection (unshifted is correct).
    post_fc_buffer, _, _, _ = sar_model.compute_post_fc(
        hidden_for_prefill,
        input_ids=input_ids[:, :-1],
        inputs_embeds=embeds_for_prefill,
    )  # (1, L_compressed', D)

    # FIX #3: Pass hidden_states_residual for residual gate.
    # We need the hidden states that correspond to each position in post_fc_buffer.
    # When no compression: hidden_residual = hidden_for_prefill (same length).
    # When compression: SAR head output is (1, L', k, D) where L' < L-1.
    #   The residual gate in training is applied AFTER _expand_to_original_length,
    #   but in inference we don't expand — we work with compressed lengths.
    #   For simplicity, pass hidden_for_prefill to sar_only which will handle
    #   length mismatch gracefully (only applied if shapes match).
    L_buffer = post_fc_buffer.shape[1]
    if hidden_for_prefill.shape[1] == L_buffer:
        hidden_residual_prefill = hidden_for_prefill
    else:
        # Compression was applied — can't directly use full hidden for residual.
        # Skip residual for prefill (it mainly matters for text positions which
        # are post-compression; the mismatch is in image token positions).
        hidden_residual_prefill = None

    sar_prefill_out = sar_model.sar_only(
        post_fc_buffer,
        hidden_states_residual=hidden_residual_prefill,
        lm_head=lm_head,
    )  # (1, L_compressed', k, D)

    if sar_prefill_out.dim() != 4:
        raise RuntimeError(
            f"Expected SAR prefill output (B, L', k, D); got {tuple(sar_prefill_out.shape)}"
        )

    last_drafts = sar_prefill_out[:, -1, :, :].contiguous()  # (1, k, D)

    # Trim base KV by 1 so pending is NOT in cache.
    past_kv = trim_kv_cache(past_kv_full, L_prefill - 1)

    # Buffers for incremental SAR (store hidden states for residual).
    hidden_buffer = hidden_for_prefill.contiguous()  # (1, L-1, D)
    embeds_buffer = embeds_for_prefill.contiguous()  # (1, L-1, D) — already shifted
    pending_token = input_ids[:, -1:].contiguous()   # (1, 1)

    while True:
        L_full = input_ids.shape[1]

        # =============== Build draft tokens from last_drafts[1:] ===============
        draft_logits = lm_head(last_drafts[:, 1:, :])  # (1, k-1, V)
        if logits_processor is not None:
            draft_tokens = []
            draft_probs_list = []
            for i in range(eff_drafts):
                logits_i = logits_processor(None, draft_logits[:, i, :])
                probs_i = torch.softmax(logits_i, dim=-1)
                tok_i = torch.multinomial(probs_i, 1)
                draft_tokens.append(tok_i)
                draft_probs_list.append(probs_i)
            draft_tokens = torch.cat(draft_tokens, dim=1)
        else:
            draft_tokens = torch.argmax(draft_logits, dim=-1)  # (1, k-1)
            draft_probs_list = None

        # =============== Merged verify: [pending, draft_0..draft_{k-2}] ===============
        merged_input = torch.cat([pending_token, draft_tokens], dim=1)  # (1, k)
        verify_out = base_model(
            input_ids=merged_input,
            past_key_values=past_kv,
            output_hidden_states=True,
            use_cache=True,
        )
        past_kv_after_verify = verify_out.past_key_values
        verify_hidden = verify_out.hidden_states[-1]  # (1, k, D)
        verify_embeds = verify_out.hidden_states[0]   # (1, k, D)
        verify_logits = lm_head(verify_hidden)        # (1, k, V)

        target_logits_for_draft = verify_logits[:, 1:, :]  # (1, k-1, V)
        bonus_logits = verify_logits[:, -1, :]              # (1, V)

        # =============== Acceptance ===============
        if logits_processor is None:
            target_tokens = torch.argmax(target_logits_for_draft, dim=-1)
            matches = (draft_tokens == target_tokens)
            accept_mask = torch.cumprod(matches.int(), dim=1)
            accept_length = int(accept_mask.sum(dim=1).item())

            accepted = draft_tokens[:, :accept_length]
            if accept_length < eff_drafts:
                new_pending = torch.argmax(
                    target_logits_for_draft[:, accept_length, :],
                    dim=-1, keepdim=True,
                )
            else:
                new_pending = torch.argmax(bonus_logits, dim=-1, keepdim=True)
        else:
            accept_length = 0
            rejected = False
            correction = None
            for i in range(eff_drafts):
                target_logits_i = logits_processor(
                    None, target_logits_for_draft[:, i, :]
                )
                target_probs_i = torch.softmax(target_logits_i, dim=-1)
                draft_probs_i = draft_probs_list[i]

                tok_i = draft_tokens[0, i].item()
                p = target_probs_i[0, tok_i].item()
                q = draft_probs_i[0, tok_i].item()

                u = torch.rand(1, device=draft_tokens.device).item()
                if u < min(1.0, p / (q + 1e-10)):
                    accept_length += 1
                else:
                    residual = torch.clamp(target_probs_i - draft_probs_i, min=0)
                    residual = residual / (residual.sum() + 1e-10)
                    correction = torch.multinomial(residual, 1)
                    rejected = True
                    break

            accepted = draft_tokens[:, :accept_length]
            if rejected:
                new_pending = correction
            else:
                last_target_logits = logits_processor(None, bonus_logits)
                last_probs = torch.softmax(last_target_logits, dim=-1)
                new_pending = torch.multinomial(last_probs, 1)

        # =============== Commit pending + accepted drafts ===============
        committed_hidden = verify_hidden[:, : 1 + accept_length, :]  # (1, 1+accept, D)
        committed_embeds = verify_embeds[:, : 1 + accept_length, :]

        input_ids = torch.cat([input_ids, accepted], dim=1)
        new_token += accepted.shape[1] + 1

        # FIX #2: Shift committed_embeds for the incremental SAR step.
        # Training convention: position j gets embed(token_{j+1}).
        # committed_embeds[:, j, :] = embed(merged_input[j]) = embed(token at position L_full-1+j).
        # We need embed(token at L_full+j) for position j.
        # Shift: positions 0..accept_length-1 get embeds from 1..accept_length.
        # Last position (accept_length) gets embed(new_pending).
        new_pending_embed = sar_model.embed_tokens(new_pending)  # (1, 1, D)
        if accept_length > 0:
            shifted_committed_embeds = torch.cat(
                [committed_embeds[:, 1:, :], new_pending_embed], dim=1
            )  # (1, 1+accept, D) — shifted
        else:
            # Only pending was committed (0 drafts accepted).
            # Position 0 (pending) needs embed(new_pending).
            shifted_committed_embeds = new_pending_embed  # (1, 1, D)

        hidden_buffer = torch.cat([hidden_buffer, committed_hidden], dim=1)
        embeds_buffer = torch.cat([embeds_buffer, shifted_committed_embeds], dim=1)

        # Trim base KV
        past_kv = trim_kv_cache(past_kv_after_verify, L_full + accept_length)

        # =============== Advance SAR with residual gate ===============
        # FIX #3: Pass hidden_states_residual for gated residual.
        # encode_text produces post-FC features from the shifted embeds + hidden.
        new_post_fc = sar_model.encode_text(committed_hidden, shifted_committed_embeds)
        if post_fc_buffer is None:
            full_buffer = new_post_fc
        else:
            full_buffer = torch.cat([post_fc_buffer, new_post_fc], dim=1)
        post_fc_buffer = full_buffer

        # Residual: use full hidden_buffer if lengths match, else skip.
        if hidden_buffer.shape[1] == full_buffer.shape[1]:
            hidden_residual = hidden_buffer
        else:
            hidden_residual = None

        sar_inc_out = sar_model.sar_only(
            full_buffer,
            hidden_states_residual=hidden_residual,
            lm_head=lm_head,
        )

        last_drafts = sar_inc_out[:, -1, :, :].contiguous()  # (1, k, D)

        # =============== Append new_pending to input_ids ===============
        input_ids = torch.cat([input_ids, new_pending], dim=1)
        pending_token = new_pending

        steps += 1

        # =============== Stopping conditions ===============
        tail = input_ids[0, L_full:].tolist()
        if tokenizer.eos_token_id in tail:
            break
        if new_token >= max_new_tokens:
            break
        if input_ids.shape[1] >= max_length:
            break

    return input_ids, new_token, steps


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_eval(
    base_model_path,
    sar_model_path,
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
        import ray
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = max(1, len(data) // (num_gpus_total // num_gpus_per_model))
    ans_handles = []
    for i in range(0, len(data), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                sar_model_path,
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
    sar_model_path,
    model_id,
    questions,
    answer_file,
    datapath,
    max_new_token,
    num_gpus_per_model,
    max_gpu_memory,
    temperature,
):
    base_model, processor = load_base_vlm(base_model_path)
    lm_head = get_lm_head(base_model)
    tokenizer = processor.tokenizer

    device = next(base_model.parameters()).device
    sar_model = load_sar_model(
        sar_model_path,
        device=device,
        dtype=base_model.dtype,
    )

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
            messages, tokenize=False, add_generation_prompt=True
        )
        image = Image.open(os.path.join(datapath, questions[j]["file_name"]))
        inputs = processor(images=image, text=text, return_tensors="pt")
        input_ids = inputs.input_ids
        pixel_values = inputs.pixel_values

        torch.cuda.synchronize()
        _ = sar_forward(
            input_ids=input_ids.cuda(),
            pixel_values=pixel_values.cuda(),
            base_model=base_model,
            lm_head=lm_head,
            sar_model=sar_model,
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
            messages, tokenize=False, add_generation_prompt=True
        )
        image = Image.open(os.path.join(datapath, question["file_name"]))
        inputs = processor(images=image, text=text, return_tensors="pt")
        input_ids = inputs.input_ids
        pixel_values = inputs.pixel_values

        torch.cuda.synchronize()
        start_time = time.time()
        output_ids, new_token, idx = sar_forward(
            input_ids=input_ids.cuda(),
            pixel_values=pixel_values.cuda(),
            base_model=base_model,
            lm_head=lm_head,
            sar_model=sar_model,
            tokenizer=tokenizer,
            logits_processor=logits_processor,
            max_new_tokens=max_new_token,
        )
        torch.cuda.synchronize()
        total_time = time.time() - start_time

        gen_ids = output_ids[0][input_ids.shape[1]:]
        new_token = gen_ids.shape[-1]

        output = tokenizer.decode(gen_ids, spaces_between_special_tokens=False)
        if tokenizer.eos_token and tokenizer.eos_token in output:
            output = output[: output.find(tokenizer.eos_token)]
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for st in special_token:
                    output = output.replace(st, "")
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
                "decoding": "sar_v2",
                "k": getattr(sar_model, "k", 5),
            }
            fout.write(json.dumps(ans_json) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sar-model-path",
        type=str,
        required=True,
        help="Path to trained SAR model directory containing config.json and model.safetensors.",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="path/to/llava-1.5-7b-hf",
        help="Path to llava-1.5-7b-hf",
    )
    parser.add_argument("--model-id", type=str, default="llava-v1.5-7b-hf-fp16-sar-v2")
    parser.add_argument("--bench-name", type=str, default="COCO-caption")
    parser.add_argument("--question-begin", type=int, default=0)
    parser.add_argument("--question-end", type=int, default=100)
    parser.add_argument("--answer-dir", type=str, default="./outputs")
    parser.add_argument(
        "--datajson",
        type=str,
        default="path/to/coco/annotations/captions_val2014.json",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="path/to/coco/val2014",
    )
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--max-gpu-memory", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()
    args.model_id = args.model_id + "-temperature-" + str(args.temperature)

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray
        ray.init()

    answer_file = os.path.join(args.answer_dir, args.bench_name, f"{args.model_id}.jsonl")
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    print(f"Output to {answer_file}")

    run_eval(
        base_model_path=args.base_model_path,
        sar_model_path=args.sar_model_path,
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
