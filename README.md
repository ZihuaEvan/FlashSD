# SpecFLASH

**A Latent-Guided Semi-autoregressive Speculative Decoding Framework for Efficient Multimodal Generation**

<p align="center">
  <em>ACM Multimedia 2026</em>
</p>

SpecFLASH is a speculative decoding framework tailored to **Large Multimodal Models (LMMs)**. Unlike prior speculative decoding methods that rely on text-only drafts and ignore the structure of visual inputs, SpecFLASH explicitly exploits two properties of multimodal data — **visual token redundancy** and **visual entity co-occurrence** — to build a fast yet high-quality draft model. It is **lossless**: the target model's verification step guarantees that the output distribution is identical to standard autoregressive decoding.

On LLaVA-1.5 and Qwen2.5-VL, SpecFLASH reaches up to **2.68× speed-up on video captioning** and **2.55× on visual instruction tuning**, outperforming previous speculative decoding baselines while preserving output quality.

---

## Method

SpecFLASH adds two lightweight components to the draft model.

### 1. Visual token compression
Visual sequences carry a lot of redundant, low-information-density tokens. A learnable set of `C` query vectors attends over the `N` second-to-top-layer visual features `F_V` and compresses them into `C` latent visual features (LLaVA: `576 → 64`, Qwen2.5-VL: `324 → 36`):

```
F̂_V = softmax(C · F_Vᵀ) · F_V
```

This shrinks the draft model's visual input from `N` to `C`, cutting draft-generation cost while retaining the most salient semantics.

### 2. Semi-autoregressive (SAR) head
Instead of drafting one token per forward pass, a semi-autoregressive head predicts the next `K` tokens in a **single** pass using `K` learnable slot embeddings and a block-diagonal attention mask. Following EAGLE, the draft consumes the target model's second-to-top-layer text feature `F_T` concatenated with token embeddings `E`, together with the compressed visual features `F̂_V`.

### Training objective
The draft head is trained with a regression loss (Smooth-L1 on the predicted vs. target hidden states) plus a classification loss (cross-entropy against the frozen LM head), balanced by `α = 0.1`:

```
L = Σ SmoothL1(F'_i, F_i) + α · Σ CE(P'_i, P_i)
```

The target LMM (vision encoder, projector, transformer, LM head) stays **frozen**; only the compressor and the SAR head are trained.

### Losslessness
At inference the `K` drafted tokens are verified in parallel by the target model with the standard speculative acceptance–rejection rule, so the generated sequence exactly follows the target model's distribution.

---

## Repository structure

```
flash/
├── model/                        # Model definitions
│   ├── llama_sar_new2.py         # ★ Final SpecFLASH draft: visual compressor + semi-AR head
│   ├── llama_sar.py              #   Base SpecFLASH draft (no Hydra refinement / scheduled sampling)
│   ├── cnets.py                  #   EAGLE / EAGLE-EYE autoregressive draft (baseline)
│   ├── ee_model.py               #   Speculative-decoding wrapper (EAGLE tree + SAR generate)
│   ├── modeling_llava.py         #   Target model: LLaVA-1.5 (patched HF impl.)
│   ├── modeling_qwen2_5_vl.py    #   Target model: Qwen2.5-VL (patched HF impl.)
│   ├── modeling_llama_kv.py      #   LLaMA backbone with exposed KV-cache
│   ├── kv_cache.py               #   KV-cache utilities for the speculative loop
│   ├── configs.py                #   Draft-model config (EConfig)
│   ├── utils.py / utils_c.py     #   Sampling, tree buffers, decoding helpers
│   └── choices.py                #   Tree-choice presets
├── ge_data/                      # Offline feature/data generation for training
│   ├── get_data_all_llava.py         # Visual instruction tuning — LLaVA
│   ├── get_data_all_qwen2.5vl.py     # Visual instruction tuning — Qwen2.5-VL
│   ├── get_data_video_all_llava.py   # Video captioning — LLaVA
│   └── get_data_video_all_qwen2.5vl.py  # Video captioning — Qwen2.5-VL
├── train/                        # Training entry points
│   ├── train_sar_new2.py         # ★ Train final SpecFLASH draft
│   ├── train_sar.py              #   Train base SpecFLASH draft
│   ├── train_llava.py            #   Train EAGLE draft on LLaVA (baseline)
│   ├── train_qwenvl2.5.py        #   Train EAGLE draft on Qwen2.5-VL (baseline)
│   ├── nar_config.json           #   SAR draft config (k, compress_len, SAR heads)
│   ├── llava_7B_config.json      #   LLaVA target config
│   ├── qwenvl2_5_7B_config.json  #   Qwen2.5-VL target config
│   └── fsdp.yaml                 #   Accelerate/FSDP launch config
└── evaluation/                   # Inference & speed measurement
    ├── gen_sar_answer_llava_v2.py    # ★ SpecFLASH inference (LLaVA)
    ├── gen_sar_answer_llava.py       #   SpecFLASH inference (base variant)
    ├── gen_ee_answer_llava1.5.py     #   EAGLE speculative baseline (LLaVA)
    ├── gen_ee_answer_qwen2.5vl.py    #   EAGLE speculative baseline (Qwen2.5-VL)
    ├── gen_baseline_answer_llava.py      # Vanilla autoregressive (speed-up denominator)
    ├── gen_baseline_answer_qwen2.5vl.py  # Vanilla autoregressive (Qwen2.5-VL)
    ├── gen_baseline_video*.py            # Vanilla video-captioning baselines
    ├── instruct_*_baseline.py            # Vanilla instruction baselines
    ├── video_qwen2_baseline.py           # Vanilla Qwen video baseline
    ├── merge_weight.py               #   Merge sharded draft checkpoints
    └── speed.py                      #   Compute speed-up ratio from two output files
```

★ = the components used for SpecFLASH in the paper. The `*_new2` / `*_v2` files are the final versions (they add a Hydra-style slot refinement and scheduled sampling on top of the base variant).

---

## Installation

```bash
pip install -e .
pip install -r requirements.txt
```

Tested with `torch==2.2.1` and `transformers==4.51.1`. A single high-memory GPU (e.g. NVIDIA A6000) is sufficient for training and evaluation.

You will need the **target model weights** (downloaded separately from the Hugging Face Hub):

- [`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

> All paths in this repository are placeholders of the form `path/to/...`. Replace them (via CLI flags or the constants at the top of each script) with your own locations for model weights, datasets, features and checkpoints.

---

## 1. Data preparation

Training the draft model is done on **offline features** dumped from the frozen target model (hidden states, input embeddings, `input_ids`, and a loss mask). Generate them with the `ge_data/` scripts.

**Datasets**
- Visual instruction tuning: [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) + COCO `train2017` images.
- Video captioning: [Kinetics-400](https://github.com/cvdfoundation/kinetics-dataset).

Set the target-model path and dataset paths at the top of each script (`bigname`, `data`, `data_files`, `video_dir`, …), then run:

```bash
# Visual instruction tuning
python -m flash.ge_data.get_data_all_llava        --outdir path/to/features/llava_vit   --gpu_index 0
python -m flash.ge_data.get_data_all_qwen2.5vl    --outdir path/to/features/qwen_vit    --gpu_index 0

# Video captioning
python -m flash.ge_data.get_data_video_all_llava      --outdir path/to/features/llava_vc
python -m flash.ge_data.get_data_video_all_qwen2.5vl  --outdir path/to/features/qwen_vc
```

`--start` / `--end` / `--index` / `--gpu_index` control sharding across GPUs.

---

## 2. Training

The SAR draft configuration lives in `flash/train/nar_config.json`:

```jsonc
{
  "k": 4,                       // number of tokens drafted per forward pass (K)
  "compress_len": 64,           // compressed visual tokens (C)
  "image_token_per_image": 576, // original visual tokens (N)
  "sar_num_heads": 64,
  "sar_head_dim": 64,
  "n_layers": 1
}
```

**Train the SpecFLASH draft (recommended, final version):**

```bash
accelerate launch --config_file flash/train/fsdp.yaml \
  flash/train/train_sar_new2.py \
  --basepath   path/to/llava-1.5-7b-hf \
  --configpath flash/train/nar_config.json \
  --tmpdir     path/to/features/llava_vit \
  --cpdir      path/to/checkpoints \
  --outdir     path/to/checkpoints \
  --lr 1e-4 --bs 2 --gradient-accumulation-steps 8
```

Use `train_sar.py` for the base (non-refined) draft, and `train_llava.py` / `train_qwenvl2.5.py` to train the EAGLE draft baselines.

---

## 3. Evaluation

Generate answers with the SpecFLASH draft, then compute the speed-up ratio against the vanilla autoregressive run.

**SpecFLASH (LLaVA):**

```bash
python -m flash.evaluation.gen_sar_answer_llava_v2 \
  --sar-model-path  path/to/checkpoints/state_x \
  --base-model-path path/to/llava-1.5-7b-hf \
  --datajson path/to/coco/annotations/captions_val2014.json \
  --datapath path/to/coco/val2014 \
  --answer-dir path/to/outputs \
  --temperature 0.0
```

**Baselines** — EAGLE speculative decoding and vanilla autoregressive (the latter is the denominator of the speed-up ratio):

```bash
# EAGLE speculative baseline
python -m flash.evaluation.gen_ee_answer_llava1.5 \
  --ee-model-path   path/to/eagle_checkpoint \
  --base-model-path path/to/llava-1.5-7b-hf

# Vanilla autoregressive target model
python -m flash.evaluation.gen_baseline_answer_llava \
  --base-model-path path/to/llava-1.5-7b-hf
```

Qwen2.5-VL has matching `*_qwen2.5vl.py` scripts.

**Speed-up ratio.** Set `jsonl_file` (your method) and `jsonl_file_base` (vanilla baseline) at the top of `speed.py`, then:

```bash
python -m flash.evaluation.speed
```

It reports the average accepted tokens `A` and the wall-clock speed-up ratio `R`.

---

## Results

Speed-up ratio `R` and average accepted tokens `A` at greedy decoding (`τ = 0`), from the paper:

| Task                        | Target model | `A`  | `R`      |
|-----------------------------|--------------|------|----------|
| Video captioning            | LLaVA-1.5    | 2.63 | **1.83×** |
| Video captioning            | Qwen2.5-VL   | 3.21 | **2.68×** |
| Visual instruction tuning   | LLaVA-1.5    | 2.77 | **2.55×** |
| Visual instruction tuning   | Qwen2.5-VL   | 2.46 | **1.83×** |

SpecFLASH consistently achieves the highest speed-up among competing speculative decoding methods (Eagle, Medusa, SpecVLM, Dream, text-only) while producing output identical to the target model.

---

## Citation

```bibtex
@inproceedings{wang2026specflash,
  title     = {SpecFLASH: A Latent-Guided Semi-autoregressive Speculative Decoding Framework for Efficient Multimodal Generation},
  author    = {Wang, Zihua and Li, Ruibo and Du, Haozhe and Zhou, Joey Tianyi and Zhang, Yu and Yang, Xu},
  booktitle = {Proceedings of the 35th ACM International Conference on Multimedia (MM '26)},
  year      = {2026},
  doi       = {10.1145/3767308.3835955}
}
```

## Acknowledgements

This project builds on [EAGLE](https://github.com/SafeAILab/EAGLE), [LLaVA](https://github.com/haotian-liu/LLaVA), and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL). We thank the authors for releasing their code and models.
