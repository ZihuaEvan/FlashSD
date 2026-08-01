"""
SAR (Semi-AutoRegressive) speculative decoding evaluation for LLaVA-1.5.

Pure SAR evaluation: loads ONLY the base VLM and the user-trained SAR draft head.
No AR (cnets) draft, no EAGLE-EYE checkpoint involvement.

Differs from gen_ee_answer_llava1.5.py:
  - Uses Semi-AR draft model (llama_sar.Model) instead of tree-based AR draft (cnets.Model)
  - One forward pass of SAR head produces k future hidden states for each position;
    we take the LAST position's k predictions, decode with frozen LM head -> k draft tokens
  - Target model verifies k tokens in a single parallel forward
  - Greedy or speculative-sampling acceptance, then advance by accepted_len + 1

Trick used to bypass `rearrange_block_diag` in inference:
  We call `sar_model.train()` while wrapped in `torch.inference_mode()`.
  In llama_sar.Model.forward(), `self.training` triggers the early-return branch that
  yields raw `(B, L', k, D)` instead of the block-diagonal `(B, L'*k, D)`.
  This makes "extract last position's k predictions" trivially `output[:, -1, :, :]`.
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
from flash.model.llama_sar import Model as SARModel
from flash.model.configs import EConfig
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Loaders — base VLM and SAR draft are loaded INDEPENDENTLY
# ---------------------------------------------------------------------------
def load_base_vlm(base_model_path, device_map="cuda:0"):
    """Load LLaVA base model + processor only. No AR draft."""
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
    # LlavaForConditionalGeneration exposes `lm_head` directly (delegates to
    # language_model.lm_head). Fall back to language_model.lm_head if needed.
    if hasattr(base_model, "lm_head") and base_model.lm_head is not None:
        return base_model.lm_head
    return base_model.language_model.lm_head


def load_sar_model(sar_model_path, device, dtype):
    """Instantiate SARModel from a folder with config.json + weights.

    Supports both ``model.safetensors`` (preferred) and ``pytorch_model.bin``
    (fallback). The training script saves ``pytorch_model.bin`` via
    ``torch.save()``, so the fallback is necessary for user-trained SAR
    checkpoints.
    """
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
    sar_model.eval()
    return sar_model


# ---------------------------------------------------------------------------
# KV cache helpers
# ---------------------------------------------------------------------------
def trim_kv_cache(cache, target_len):
    """Truncate past_key_values along the sequence dim to ``target_len``.

    Supports both transformers ``DynamicCache`` (>=4.36) and the legacy
    tuple-of-tuples format.  Cache tensors have shape
    ``(B, num_heads, seq_len, head_dim)``.
    """
    if cache is None:
        return None

    # Modern Cache objects (DynamicCache, etc.)
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

    # Legacy tuple-of-tuples
    if isinstance(cache, tuple):
        return tuple(
            (k[..., :target_len, :], v[..., :target_len, :])
            for (k, v) in cache
        )

    raise TypeError(f"Unknown past_key_values type: {type(cache)}")


# ---------------------------------------------------------------------------
# SAR speculative decoding loop WITH KV cache (base + SAR head)
# AND merged extra/correction forward into next-iter verify.
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
    """SAR speculative decoding (Design A: block-diagonal placeholder parallel).

    Optimizations layered on:

    1. Base-model KV cache: verification forward only processes k new tokens
       (1 pending + k-1 drafts) instead of the full L+k sequence.
    2. Post-FC buffer reuse: the SAR head's full attention is over (L'*k)
       tokens with a block-diagonal causal mask, so it cannot truly cache K/V
       across calls.  We instead cache the *post-FC features* (compression +
       FC output) and append the new chunk per iter, then re-run the SAR
       head over the full updated buffer.  The expensive visual compression
       runs only once at prefill.
    3. Merged pending+drafts verify: the "pending" token (either the last
       prompt token at iter 1, or the correction/bonus token from iter t-1)
       is concatenated with iter-t draft tokens into a single k-token
       base-model forward.  This eliminates the standalone 1-token "extra"
       forward (saves one kernel launch per iter).

    Tradeoff: merging consumes SAR's draft[0] slot (it lands at pending's
    position which is already decided).  Effective drafts per iter = k-1.

    Position bookkeeping (iter t, going in):
        input_ids       length = L_full   ; last token is ``pending`` (no hidden yet)
        past_kv         length = L_full - 1   (does NOT include pending)
        hidden_buffer   length = L_full - 1
        post_fc_buffer  length = L_compressed' (does NOT include pending)
        last_drafts     shape  (1, k, D)  ; SAR's k drafts; slot 0 lands at
                                            pending pos (unused); slots 1..k-1
                                            predict L_full..L_full+k-2
    """
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    input_ids = input_ids.clone()

    k = getattr(sar_model, "k", 5)
    if k < 2:
        raise ValueError(f"Merged-verify SAR requires k>=2; got k={k}")
    eff_drafts = k - 1   # SAR's draft[0] slot is consumed by pending merging
    new_token = 0
    steps = 0

    # SAR forward branches on `self.training` to skip rearrange_block_diag —
    # but we use `sar_only` directly here, which already returns raw
    # (B, L', k, D), so the train() flip is not strictly required.
    # We keep it for safety in case future changes route through Model.forward().
    prev_training = sar_model.training
    sar_model.train()
    sar_model.reset_cache()

    try:
        # ================== Initial prefill (uses image) ==================
        prefill_out = base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            output_hidden_states=True,
            use_cache=True,
        )
        past_kv_full = prefill_out.past_key_values            # length L_prefill
        hidden_prefill = prefill_out.hidden_states[-1]        # (1, L_prefill, D)
        embeds_prefill = prefill_out.hidden_states[0]         # (1, L_prefill, D)
        L_prefill = input_ids.shape[1]

        # SAR prefill (Design A): compute post-FC buffer ONCE (compression +
        # FC over all prefix-but-pending tokens), then run the SAR head over
        # that buffer to get the initial drafts.  We deliberately EXCLUDE the
        # last prompt token (= pending) so the post_fc buffer covers
        # everything except pending — matching the merged-verify invariant.
        post_fc_buffer, _, _, _ = sar_model.compute_post_fc(
            hidden_prefill[:, :-1, :],
            input_ids=input_ids[:, :-1],
            inputs_embeds=embeds_prefill[:, :-1, :],
        )  # (1, L_compressed', D)
        sar_prefill_out = sar_model.sar_only(post_fc_buffer)  # (1, L_compressed', k, D)
        if sar_prefill_out.dim() != 4:
            raise RuntimeError(
                f"Expected SAR prefill output (B, L', k, D); got {tuple(sar_prefill_out.shape)}"
            )
        # last_drafts: SAR's drafts at the LAST post-FC position (= base pos L_prefill-2).
        # Its k drafts predict base positions L_prefill-1..L_prefill+k-2.
        #   slot 0 -> L_prefill-1 = pending (known) -> DISCARDED.
        #   slots 1..k-1 -> L_prefill..L_prefill+k-2 -> our (k-1) drafts.
        last_drafts = sar_prefill_out[:, -1, :, :].contiguous()   # (1, k, D)

        # Trim base KV by 1 so pending is NOT in cache; iter 1's merged verify
        # will re-introduce it as the first token.
        past_kv = trim_kv_cache(past_kv_full, L_prefill - 1)
        hidden_buffer = hidden_prefill[:, :-1, :].contiguous()
        embeds_buffer = embeds_prefill[:, :-1, :].contiguous()
        pending_token = input_ids[:, -1:].contiguous()            # (1, 1)

        while True:
            L_full = input_ids.shape[1]   # length INCLUDING pending

            # =============== Build draft tokens from last_drafts[1:] ===============
            # Skip slot 0 (lands at pending's position, already known).
            draft_logits = lm_head(last_drafts[:, 1:, :])     # (1, k-1, V)
            if logits_processor is not None:
                draft_tokens = []
                draft_probs_list = []
                for i in range(eff_drafts):
                    logits_i = logits_processor(None, draft_logits[:, i, :])
                    probs_i = torch.softmax(logits_i, dim=-1)
                    tok_i = torch.multinomial(probs_i, 1)
                    draft_tokens.append(tok_i)
                    draft_probs_list.append(probs_i)
                draft_tokens = torch.cat(draft_tokens, dim=1)  # (1, k-1)
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
            past_kv_after_verify = verify_out.past_key_values  # length L_full - 1 + k
            verify_hidden = verify_out.hidden_states[-1]       # (1, k, D)
            verify_embeds = verify_out.hidden_states[0]        # (1, k, D)
            verify_logits = lm_head(verify_hidden)             # (1, k, V)

            # verify_logits[:, j, :] = base-model prediction at position (L_full-1)+j.
            #   j=0:    position L_full-1 = pending (already known)        -- unused
            #   j=i+1:  position L_full + i = supervises draft_i           (i=0..k-2)
            # So target_logits_for_draft = verify_logits[:, 1:, :] (length k-1).
            # Bonus logit (if all k-1 drafts accepted): verify_logits[:, k-1, :].
            target_logits_for_draft = verify_logits[:, 1:, :]             # (1, k-1, V)
            bonus_logits = verify_logits[:, -1, :]                         # (1, V)

            # =============== Acceptance ===============
            if logits_processor is None:
                # Greedy
                target_tokens = torch.argmax(target_logits_for_draft, dim=-1)  # (1, k-1)
                matches = (draft_tokens == target_tokens)
                accept_mask = torch.cumprod(matches.int(), dim=1)
                accept_length = int(accept_mask.sum(dim=1).item())  # 0..k-1

                accepted = draft_tokens[:, :accept_length]
                if accept_length < eff_drafts:
                    new_pending = torch.argmax(
                        target_logits_for_draft[:, accept_length, :],
                        dim=-1, keepdim=True,
                    )  # (1, 1) correction
                else:
                    new_pending = torch.argmax(bonus_logits, dim=-1, keepdim=True)  # bonus
            else:
                # Speculative sampling
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
            # Positions L_full-1 (pending) and L_full..L_full-1+accept_length (drafts).
            # Their hidden states are verify_hidden[:, 0..accept_length, :].
            committed_hidden = verify_hidden[:, : 1 + accept_length, :]   # (1, 1+accept, D)
            committed_embeds = verify_embeds[:, : 1 + accept_length, :]

            input_ids = torch.cat([input_ids, accepted], dim=1)
            new_token += accepted.shape[1] + 1  # +1 for new_pending appended below

            hidden_buffer = torch.cat([hidden_buffer, committed_hidden], dim=1)
            embeds_buffer = torch.cat([embeds_buffer, committed_embeds], dim=1)

            # Trim base KV: past_kv_after_verify length = L_full - 1 + k.
            # Keep prefix length L_full + accept_length (pending + accepted).
            past_kv = trim_kv_cache(past_kv_after_verify, L_full + accept_length)

            # =============== Advance SAR over the updated post-FC buffer ===============
            # Design A cannot incrementally cache attention K/V (block-diagonal
            # mask spans full sequence), but it can cache the *post-FC features*.
            # forward_incremental encodes the new chunk through FC, appends to
            # the running post_fc_buffer, then re-runs the SAR head over the
            # full updated buffer.  Returns (sar_out, new_post_fc_buffer).
            sar_inc_out, post_fc_buffer = sar_model.forward_incremental(
                committed_hidden, committed_embeds, post_fc_buffer=post_fc_buffer,
            )
            # sar_inc_out: (1, L_compressed_new', k, D); LAST position's k drafts predict
            # positions L_full+accept_length..L_full+accept_length+k-1.
            # Slot 0 lands at new_pending's pos (= L_full+accept_length) — that's
            # the pending of NEXT iter, so we discard slot 0 again next iter.
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
    finally:
        sar_model.train(prev_training)
        sar_model.reset_cache()

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
    # ----- load base VLM (no AR draft) -----
    base_model, processor = load_base_vlm(base_model_path)
    lm_head = get_lm_head(base_model)
    tokenizer = processor.tokenizer

    # ----- load SAR draft head from user's trained checkpoint -----
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
                "decoding": "sar",
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
    parser.add_argument("--model-id", type=str, default="llava-v1.5-7b-hf-fp16-sar")
    parser.add_argument("--bench-name", type=str, default="COCO-caption")
    parser.add_argument("--question-begin", type=int, default=0)
    parser.add_argument("--question-end", type=int, default=100)
    parser.add_argument("--answer-dir", type=str, default="./outputs")
    parser.add_argument(
        "--datajson",
        type=str,
        default="path/to/coco/annotations/captions_val2014.json",
        help="Path to the input JSON file containing questions or data.",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="path/to/coco/val2014",
        help="Name or path of the dataset to be used for evaluation.",
    )
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--max-gpu-memory", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)

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
