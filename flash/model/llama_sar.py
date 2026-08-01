import torch
import torch.nn as nn
from einops import rearrange
import copy
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

try:
    from .configs import EConfig
    from .utils_c import *
    from .choices import *
    #from .utils_patch import random_mask_delete
except:
    from configs import EConfig
    from utils_c import *
    from choices import *
    from utils import prepare_logits_processor
    #from utils_patch import random_mask_delete

def rearrange_block_diag(x, pad_index=None):
    """
    把 (B, L*k, D) 的 x 按 “偏移” 重新排列，并且对越界（j + i >= L）的都填 pad。
    最终输出依然是 (B, L*k, D)，但顺序如示例。

    参数:
      x: Tensor, shape (B, L*k, D)
      L: int, 原始序列长度
      k: int, 窗口/偏移 数
      pad_embedding: Tensor of shape (D,), or None 用 0 填充
    """
    B, L, k, D = x.shape
    device = x.device

    # pad embedding


    # (B, L, k, D) → (B, k, L, D)
    x2 = x.view(B, L, k, D)
    x3 = x2.transpose(1, 2)  # now x3[b,i,j] == x[b, j*k + i]
    if pad_index is None:
        pad_embedding = torch.zeros(D, device=device, dtype=x3.dtype)
    else:
        pad_embedding = torch.ones(D, device=device, dtype=x3.dtype)*pad_index
    # 初始化全 pad 的输出
    out = pad_embedding.view(1, 1, D).expand(B, L * k, D).clone()

    # 逐偏移写入
    for i in range(k):
        max_j = L - i
        if max_j <= 0:
            break
        start = i * (L + 1)
        idx = start + torch.arange(max_j, device=device)  # 位置列表
        # 把 x3[:, i, 0:max_j] 批量写进去
        out[:, idx, :] = x3[:, i, :max_j, :]

    return out


def rotate_half(x):
    # 修正维度拆分问题，确保最后一个维度是偶数
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 确保维度是偶数
        assert dim % 2 == 0, "旋转维度必须是偶数"
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def _apply_rotary(self, x, cos, sin):
        # 添加维度检查
        assert x.size(-1) % 2 == 0, f"最后维度大小必须是偶数，当前为{x.size(-1)}"
        return x * cos + rotate_half(x) * sin

    def forward(self, q, k, cache_position=None, k_position=None):
        # q形状: [batch, num_heads, seq_len_q, head_dim]
        # k形状: [batch, group_size, seq_len_k, head_dim]
        # cache_position: q 的绝对位置序列。None 表示从 0 开始。
        # k_position:     k 的绝对位置序列。None 表示从 0 开始（旧行为）。
        device = q.device
        q_len = q.size(2)
        k_len = k.size(2)

        if cache_position is None:
            t_q = torch.arange(q_len, device=device).type_as(self.inv_freq)
        else:
            t_q = cache_position.to(device=device, dtype=self.inv_freq.dtype)
        if k_position is None:
            t_k = torch.arange(k_len, device=device).type_as(self.inv_freq)
        else:
            t_k = k_position.to(device=device, dtype=self.inv_freq.dtype)

        freqs_q = torch.einsum('i,j->ij', t_q, self.inv_freq)
        freqs_k = torch.einsum('i,j->ij', t_k, self.inv_freq)
        emb_q = torch.cat([freqs_q, freqs_q], dim=-1)
        emb_k = torch.cat([freqs_k, freqs_k], dim=-1)

        # 扩展维度用于广播
        cos_q = emb_q.cos()[None, None, :, None, :]
        sin_q = emb_q.sin()[None, None, :, None, :]
        cos_k = emb_k.cos()[None, None, :, None, :]
        sin_k = emb_k.sin()[None, None, :, None, :]

        # 应用旋转到每个头
        q_rot = self._apply_rotary(q.unsqueeze(-2), cos_q, sin_q).squeeze(-2)
        k_rot = self._apply_rotary(k.unsqueeze(-2), cos_k, sin_k).squeeze(-2)

        return q_rot, k_rot


class LlamaRMSNorm(nn.Module):
    """RMSNorm matching the formulation used in Llama."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaSemiARHead(nn.Module):
    """Block-diagonal placeholder parallel decoder (Design A).

    For each base position t we expand the input to k tokens by adding
    learnable slot embeddings:

        block[t][0] = h_t                       # real (slot 0 is identity)
        block[t][i] = h_t + slot_embeddings[i]  # placeholder, i = 1..k-1

    Concatenated in block-diagonal order: (B, L*k, D).

    Rotary positions:
        token (t, i) -> position t + i.
        Real h_t (slot 0) gets position t; placeholder slot i in block t
        gets position t + i.  This way each placeholder sits at the
        position of the token it's trying to predict.

    Block-diagonal causal attention.  Token (t, i) attends to:
        (t', 0)  for t' < t           # past real tokens (slot 0 only)
        (t,  j)  for j <= i           # own block prefix (incl. slot 0 = real h_t)

    GQA: num_heads query heads share group_size KV heads
    (each KV head serves num_heads / group_size query heads — standard GQA).

    Output: (B, L, k, D) where output[:, t, i, :] predicts token at t+i+1.
    """

    def __init__(self, dim, k, num_heads=8, head_dim=128, group_size=4):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even (rotary)"

        self.dim = dim
        self.attn_dim = num_heads * head_dim
        self.k = k
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.group_size = num_heads
        self.gqa_expand = 1

        intermediate_dim = 4 * dim

        self.slot_embeddings = nn.Parameter(torch.zeros(k, dim))
        with torch.no_grad():
            nn.init.normal_(self.slot_embeddings[1:], std=0.02)

        self.input_layernorm = LlamaRMSNorm(dim, eps=1e-6)
        self.post_attn_layernorm = LlamaRMSNorm(dim, eps=1e-6)

        # Full MHA — attn_dim can be wider than dim for more capacity.
        self.q_proj = nn.Linear(dim, self.attn_dim)
        self.k_proj = nn.Linear(dim, self.attn_dim)
        self.v_proj = nn.Linear(dim, self.attn_dim)
        self.o_proj = nn.Linear(self.attn_dim, dim)

        self.rotary_emb = RotaryEmbedding(head_dim)

        self.gate_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, dim, bias=False)

        self.attn_dropout = nn.Dropout(0.1)
        self.ffn_dropout = nn.Dropout(0.1)

        # Gradient checkpointing: saves VRAM by recomputing activations during backward.
        # Set to True by Model.__init__; disabled during inference.
        self.gradient_checkpointing = False

        # Mask cache to avoid rebuilding per call (keyed on (L, k, device str)).
        self._mask_cache = {}

        # KV cache stubs (Design A recomputes attention each call; kept for API compat).
        self.k_cache = None
        self.v_cache = None
        self._cache_seqlen = 0

    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None
        self._cache_seqlen = 0

    def trim_cache(self, target_len):
        # No-op in Design A — full SAR forward each inference call.
        pass

    def _build_block_diag_mask(self, L, k, device, padding_mask=None):
        """Block-diagonal causal mask of shape (L*k, L*k).
        True = blocked.  Token (t_q, i_q) at flat idx t_q*k+i_q attends to
        (t_kv, j_kv) iff:
            (t_kv < t_q AND j_kv == 0)  OR  (t_kv == t_q AND j_kv <= i_q)

        Within the same block, attention is causal (slot i sees slots 0..i).
        Across blocks, only slot 0 (the real hidden state) is visible.

        If padding_mask is provided (B, L), positions where padding_mask==0
        are blocked from being attended to.
        """
        cache_key = (L, k, str(device))
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            base_mask = cached
        else:
            idx = torch.arange(L * k, device=device)
            t_idx = idx // k
            i_idx = idx % k
            t_q = t_idx[:, None]
            i_q = i_idx[:, None]
            t_kv = t_idx[None, :]
            j_kv = i_idx[None, :]
            allowed = ((t_kv < t_q) & (j_kv == 0)) | ((t_kv == t_q) & (j_kv <= i_q))
            base_mask = ~allowed  # True = blocked
            self._mask_cache[cache_key] = base_mask

        if padding_mask is None:
            return base_mask

        B = padding_mask.shape[0]
        pad_expanded = padding_mask.repeat_interleave(k, dim=1)  # (B, L*k)
        pad_blocked = (pad_expanded[:, None, :] == 0)  # (B, 1, L*k) — block KV at padded positions
        combined = base_mask[None, :, :] | pad_blocked  # (B, L*k, L*k)
        return combined

    def forward(self, x, use_cache=False, padding_mask=None):
        """Forward pass for Design A.

        Args:
            x: (B, L, D) post-FC features.
            use_cache: kept for API compat; ignored (no SAR-head KV cache here).
            padding_mask: (B, L) optional, 1=real token, 0=padding.

        Returns:
            (B, L, k, D) — slot i at base t predicts token at position t+i+1.
        """
        del use_cache  # API compat; ignored
        B, L, D = x.shape
        k = self.k
        Lk = L * k
        H = self.num_heads
        G = self.group_size
        Dh = self.head_dim

        # ============ 1. Expand input with slot embeddings ============
        x_expanded = x.unsqueeze(2) + self.slot_embeddings[None, None, :, :]
        x_flat = x_expanded.reshape(B, Lk, D)

        residual_in = x_flat

        # ============ 2. Pre-attn norm + QKV projections ============
        x_norm = self.input_layernorm(x_flat)
        q = self.q_proj(x_norm).view(B, Lk, H, Dh).permute(0, 2, 1, 3)
        kk = self.k_proj(x_norm).view(B, Lk, H, Dh).permute(0, 2, 1, 3)
        v  = self.v_proj(x_norm).view(B, Lk, H, Dh).permute(0, 2, 1, 3)

        # ============ 3. Rotary: position(t, i) = t + i ============
        t_idx = torch.arange(L, device=x.device).repeat_interleave(k)
        i_idx = torch.arange(k, device=x.device).repeat(L)
        pos_ids = (t_idx + i_idx).to(self.rotary_emb.inv_freq.dtype)
        q_rot, k_rot = self.rotary_emb(q, kk, cache_position=pos_ids, k_position=pos_ids)

        # ============ 4. K, V already full MHA (no GQA expansion needed) ============
        k_full = k_rot
        v_full = v

        # ============ 5. Block-diagonal attention via SDPA ============
        mask_blocked = self._build_block_diag_mask(L, k, x.device, padding_mask=padding_mask)
        attn_mask = ~mask_blocked  # True = allowed for SDPA
        if attn_mask.dim() == 2:
            attn_mask = attn_mask[None, None, :, :]
        elif attn_mask.dim() == 3:
            attn_mask = attn_mask[:, None, :, :]

        def _attn_forward(q_rot, k_full, v_full, attn_mask):
            attn_output = F.scaled_dot_product_attention(
                q_rot, k_full, v_full,
                attn_mask=attn_mask,
                dropout_p=0.0 if not self.training else 0.0,
                is_causal=False,
            )
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, Lk, H * Dh)
            attn_output = self.attn_dropout(self.o_proj(attn_output))
            return attn_output

        if self.gradient_checkpointing and self.training:
            attn_output = torch.utils.checkpoint.checkpoint(
                _attn_forward, q_rot, k_full, v_full, attn_mask,
                use_reentrant=False,
            )
        else:
            attn_output = _attn_forward(q_rot, k_full, v_full, attn_mask)

        # ============ 6. Residual after attention ============
        h1 = residual_in + attn_output

        # ============ 7. Pre-MLP norm + Llama-style FFN ============
        def _ffn_forward(h1_input):
            h1_norm = self.post_attn_layernorm(h1_input)
            return self.ffn_dropout(self.down_proj(F.silu(self.gate_proj(h1_norm)) * self.up_proj(h1_norm)))

        if self.gradient_checkpointing and self.training:
            mlp_out = torch.utils.checkpoint.checkpoint(
                _ffn_forward, h1,
                use_reentrant=False,
            )
        else:
            mlp_out = _ffn_forward(h1)

        # ============ 8. Residual after MLP ============
        output_flat = h1 + mlp_out

        # ============ 9. Reshape to (B, L, k, D) ============
        return output_flat.view(B, L, k, D)


class Model(nn.Module):
    def __init__(self, config, load_emb=False, path=None, bias=True):
        super().__init__()

        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.k = config.k

        # Image token config: default to LLaVA's 32000, override from config if available
        self.image_token_id = getattr(config, 'image_token_id', 32000)
        self.image_token_per_image = getattr(config, 'image_token_per_image', 576)
        self.compress_len = getattr(config, 'compress_len', 64)

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        # Visual Compressor: learnable query matrix for attention pooling
        self.A = nn.Parameter(torch.empty(self.compress_len, config.hidden_size))
        nn.init.normal_(self.A, std=0.02)

        # Gated FC fusion: SiLU(gate) * up, more expressive than linear FC
        self.fc_gate = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias)
        self.fc_up = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias)

        # Per-slot residual gate: different slots need different residual strength
        self.residual_gate = nn.Parameter(torch.ones(config.k))

        # Semi-AR Head: predicts k future hidden states
        # head_dim fixed at 128 for RoPE; num_heads controls total attention capacity.
        sar_head_dim = getattr(config, 'sar_head_dim', 128)
        sar_num_heads = getattr(config, 'sar_num_heads', config.num_attention_heads)
        self.layers = LlamaSemiARHead(
            dim=config.hidden_size,
            k=config.k,
            num_heads=sar_num_heads,
            head_dim=sar_head_dim,
        )

        # Enable gradient checkpointing in SAR head to save VRAM
        self.layers.gradient_checkpointing = self.gradient_checkpointing

        # Freeze embedding weights
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

        # Load pretrained embeddings if requested
        if load_emb and path is not None:
            self._load_embeddings(config, path)

    def _load_embeddings(self, config, path):
        """Load pretrained token embeddings from the base model."""
        from safetensors import safe_open
        import json

        Type = config.architectures[0] if hasattr(config, 'architectures') else "LlavaForConditionalGeneration"

        if Type == "LlavaForConditionalGeneration":
            emb_key = "language_model.model.embed_tokens.weight"
        elif Type == "Qwen2_5_VLForConditionalGeneration":
            emb_key = "model.embed_tokens.weight"
        else:
            emb_key = "model.embed_tokens.weight"

        try:
            with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                index_json = json.loads(f.read())
                emb_path = index_json["weight_map"][emb_key]
            with safe_open(
                os.path.join(path, emb_path), framework="pt", device="cpu"
            ) as f:
                tensor_slice = f.get_slice(emb_key)
                vocab_size, hidden_dim = tensor_slice.get_shape()
                tensor = tensor_slice[:, :hidden_dim].float()
        except Exception:
            try:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"][emb_key]
                weights = torch.load(os.path.join(path, emb_path), map_location="cpu")
                tensor = weights[emb_key].float()
            except Exception:
                print(f"Warning: Could not load embeddings from {path}")
                return

        self.embed_tokens.weight.data = tensor

    # ------------------------------------------------------------------
    # KV-cache passthroughs (delegate to LlamaSemiARHead)
    # ------------------------------------------------------------------
    def reset_cache(self):
        """Drop the SAR head's K/V cache. Call once per decoding session."""
        if hasattr(self.layers, "reset_cache"):
            self.layers.reset_cache()

    def trim_cache(self, target_len):
        """Truncate the SAR head's K/V cache along the seq dim."""
        if hasattr(self.layers, "trim_cache"):
            self.layers.trim_cache(target_len)

    @property
    def cache_seqlen(self):
        return getattr(self.layers, "_cache_seqlen", 0)

    def _gated_fc(self, x):
        """Gated FC fusion: SiLU(gate(x)) * up(x)."""
        return F.silu(self.fc_gate(x)) * self.fc_up(x)

    def encode_text(self, hidden_states_new, inputs_embeds_new):
        """Encode post-prefill text tokens through the gated FC layer.

        Args:
            hidden_states_new: (B, L_new, D) — base-model hidden states for new tokens.
            inputs_embeds_new: (B, L_new, D) — base-model token embeddings for new tokens.

        Returns:
            (B, L_new, D) — post-FC features ready to be appended to the SAR buffer.
        """
        hidden_states_new = hidden_states_new.to(self.layers.q_proj.weight.dtype)
        inputs_embeds_new = inputs_embeds_new.to(hidden_states_new.dtype)
        return self._gated_fc(torch.cat((inputs_embeds_new, hidden_states_new), dim=-1))

    def sar_only(self, post_fc_buffer, padding_mask=None, hidden_states_residual=None):
        """Run only the SAR head over a post-FC buffer.

        Args:
            post_fc_buffer: (B, L', D) — accumulated post-FC features.
            padding_mask: (B, L') optional, 1=real token, 0=padding.
            hidden_states_residual: (B, L', D) optional — raw target hidden
                states for residual skip connection. If provided, output
                becomes SAR_head_output + hidden_states_residual.

        Returns:
            (B, L', k, D) — predictions for token at base-position t+i+1.
        """
        out = self.layers(post_fc_buffer, padding_mask=padding_mask)
        if hidden_states_residual is not None:
            gate = self.residual_gate[None, None, :, None]
            out = out + gate * hidden_states_residual.unsqueeze(2)
        return out

    def forward_incremental(self, hidden_states_new, inputs_embeds_new,
                            post_fc_buffer=None):
        """Compatibility wrapper for incremental SAR inference under Design A.

        Design A's block-diagonal attention cannot truly use a per-call
        K/V cache, so "incremental" here just means: encode the new tokens
        through FC, append to the running ``post_fc_buffer`` (caller-managed),
        and re-run the SAR head over the full updated buffer.

        Args:
            hidden_states_new: (B, L_new, D) — new base-model hidden states.
            inputs_embeds_new: (B, L_new, D) — new base-model embeddings.
            post_fc_buffer: (B, L_old, D) or None — running post-FC buffer
                from previous calls.  If None, treated as an empty buffer.

        Returns:
            (sar_output, new_post_fc_buffer)
              sar_output:        (B, L_old + L_new, k, D) — full SAR output
                                 over the updated buffer.
              new_post_fc_buffer: (B, L_old + L_new, D) — caller should
                                  pass this back on the next call.
        """
        new_post_fc = self.encode_text(hidden_states_new, inputs_embeds_new)
        if post_fc_buffer is None:
            full_buffer = new_post_fc
        else:
            full_buffer = torch.cat([post_fc_buffer, new_post_fc], dim=1)
        sar_out = self.sar_only(full_buffer)
        return sar_out, full_buffer


    def _compress_variable_length(
        self, hidden_states, inputs_embeds, mid_mask, pre_mask, post_mask,
        mid_lengths, batch_size, dim, n_img
    ):
        """Per-sample compression with padding for variable image token counts."""
        combined_list = []
        for b in range(batch_size):
            if mid_lengths[b].item() <= 0:
                combined_b = self._gated_fc(torch.cat(
                    (inputs_embeds[b], hidden_states[b]), dim=-1
                ))
            else:
                sample_mid = hidden_states[b, mid_mask[b]]
                pre_b = hidden_states[b, pre_mask[b]]
                post_emb_b = inputs_embeds[b, post_mask[b]]
                post_hid_b = hidden_states[b, post_mask[b]]

                tokens_per_img = mid_lengths[b].item() // max(n_img, 1)
                if tokens_per_img <= 0:
                    tokens_per_img = mid_lengths[b].item()
                    n_img_this = 1
                else:
                    n_img_this = max(mid_lengths[b].item() // tokens_per_img, 1)

                img_feats_b = sample_mid.view(n_img_this, tokens_per_img, dim)
                A = self.A.to(dtype=hidden_states.dtype, device=hidden_states.device)
                attn_scores = torch.einsum("h d, n t d -> n h t", A, img_feats_b)
                attn_weights = torch.softmax(attn_scores, dim=-1)
                compressed_b = torch.einsum("n h t, n t d -> n h d", attn_weights, img_feats_b)
                compressed_b = compressed_b.view(-1, dim)

                post_fc_b = self._gated_fc(torch.cat((post_emb_b, post_hid_b), dim=-1))
                combined_b = torch.cat([pre_b, compressed_b, post_fc_b], dim=0)

            combined_list.append(combined_b)

        max_len = max(c.shape[0] for c in combined_list)
        padded = []
        for c in combined_list:
            if c.shape[0] < max_len:
                pad = torch.zeros(max_len - c.shape[0], dim, dtype=c.dtype, device=c.device)
                c = torch.cat([c, pad], dim=0)
            padded.append(c)
        return torch.stack(padded, dim=0)

    def _expand_to_original_length(self, output, start_indices, mid_lengths, original_seq_len):
        """
        Expand compressed model output (B, L', k, D) back to original sequence length (B, L, k, D).

        After visual compression, L' = L - (image_tokens - compress_len).  We need to insert
        zero tensors at positions [P+compress_len, P+image_tokens) so that the expanded output
        is indexed identically to the pre-computed target_hidden from CustomDataset_sar.

        Position mapping in the expanded output:
          [0,        P)              <- copied from compressed positions [0, P)  (pre-image text)
          [P,        P+compress_len) <- copied from compressed positions [P, P+compress_len)
                                       (compressed visual, loss_mask=0 → won't affect loss)
          [P+compress_len, P+M)      <- ZEROS  (dummy, fills the "missing" image slots;
                                       these positions have loss_mask=0 in the target)
          [P+M,      L)              <- copied from compressed positions [P+compress_len, L')
                                       (post-image text / response, loss_mask=1 for responses)

        Because loss_mask is 0 for all image-token positions (they are in the instruction part),
        the dummy zeros never contribute to the regression or classification loss.
        """
        batch_size, L_prime, k, dim = output.shape
        device = output.device
        dtype = output.dtype

        expanded = torch.zeros(batch_size, original_seq_len, k, dim, dtype=dtype, device=device)

        for b in range(batch_size):
            P = int(start_indices[b].item())
            M = int(mid_lengths[b].item())
            if M <= 0:
                # No image in this sample — output is already full-length, just copy.
                copy_len = min(L_prime, original_seq_len)
                expanded[b, :copy_len] = output[b, :copy_len]
                continue

            n_images = max(M // self.image_token_per_image, 1)
            compress_total = n_images * self.compress_len   # = n_images * 64

            # 1. Pre-image text positions [0, P)
            if P > 0:
                expanded[b, :P] = output[b, :P]

            # 2. Compressed image positions [P, P+compress_total)
            #    These will be masked in the loss but we copy them for completeness.
            if compress_total > 0 and P + compress_total <= L_prime:
                expanded[b, P:P + compress_total] = output[b, P:P + compress_total]

            # 3. Dummy zeros at [P+compress_total, P+M)  — already zero from initialization.

            # 4. Post-image text positions [P+M, original_seq_len)
            #    Correspond to compressed positions [P+compress_total, L').
            post_src_start = P + compress_total
            post_dst_start = P + M
            post_len = L_prime - post_src_start
            if post_len > 0 and post_dst_start < original_seq_len:
                copy_len = min(post_len, original_seq_len - post_dst_start)
                expanded[b, post_dst_start:post_dst_start + copy_len] = \
                    output[b, post_src_start:post_src_start + copy_len]

        return expanded  # (B, L, k, D)

    def compute_post_fc(self, hidden_states, input_ids, inputs_embeds=None):
        """Run only the visual-compression + FC stages (no SAR head).

        This is the prefix of ``forward()`` that produces the input to the
        SAR head, exposed as a public method so inference code can manage
        a running post-FC buffer across decoding iterations.

        Args:
            hidden_states: (B, L, D) — second-to-top features from target model.
            input_ids:     (B, L)    — token ids (used to locate <image> tokens).
            inputs_embeds: (B, L, D) — embeddings (optional; computed from ids if None).

        Returns:
            combined:             (B, L', D) — post-compression-and-FC features.
                                  L' = L when no <image> tokens are present;
                                  L' = L - (image_tokens - compress_len) otherwise.
            compression_applied:  bool — whether visual compression actually ran.
            start_indices:        (B,) or None — image start position per sample.
            mid_lengths:          (B,) or None — image token count per sample.
        """
        batch_size, seq_length, dim = hidden_states.shape

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)

        image_mask = (input_ids == self.image_token_id)
        has_image = image_mask.any(dim=1).any()

        compression_applied = False
        start_indices = None
        mid_lengths = None

        if has_image:
            start_indices = image_mask.int().argmax(dim=1)
            mid_lengths = image_mask.sum(dim=1)

            samples_with_image = mid_lengths > 0
            if samples_with_image.sum() > 0:
                actual_tokens = mid_lengths[samples_with_image]
                unique, counts = torch.unique(actual_tokens, return_counts=True)
                canonical_token_count = unique[counts.argmax()].item()
            else:
                canonical_token_count = self.image_token_per_image

            total_mid = mid_lengths.sum().item()
            if canonical_token_count <= 0 or total_mid == 0:
                text_input = torch.cat((inputs_embeds, hidden_states), dim=-1)
                combined = self._gated_fc(text_input)
            else:
                n_img = canonical_token_count // self.image_token_per_image
                if n_img <= 0:
                    n_img = 1

                compression_applied = True

                pos = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
                pre_mask = pos < start_indices.unsqueeze(-1)
                mid_mask = (pos >= start_indices.unsqueeze(-1)) & (pos < (start_indices + mid_lengths).unsqueeze(-1))
                post_mask = pos >= (start_indices + mid_lengths).unsqueeze(-1)

                samples_with_img = [b for b in range(batch_size) if mid_lengths[b].item() > 0]
                if len(samples_with_img) == batch_size:
                    tokens_per_img_list = [mid_lengths[b].item() // max(n_img, 1) for b in range(batch_size)]
                    max_tokens_per_img = max(tokens_per_img_list)
                    if max_tokens_per_img == min(tokens_per_img_list):
                        all_mid = hidden_states[mid_mask]
                        img_feats = all_mid.view(batch_size, n_img, max_tokens_per_img, dim)
                        A = self.A.to(dtype=hidden_states.dtype, device=hidden_states.device)
                        attn_scores = torch.einsum("h d, b n t d -> b n h t", A, img_feats)
                        attn_weights = torch.softmax(attn_scores, dim=-1)
                        compressed_feats = torch.einsum("b n h t, b n t d -> b n h d", attn_weights, img_feats)
                        compressed_feats = compressed_feats.view(batch_size, -1, dim)
                        all_pre = hidden_states[pre_mask]
                        pre_feats = all_pre.view(batch_size, -1, dim)
                        all_post_emb = inputs_embeds[post_mask]
                        all_post_hid = hidden_states[post_mask]
                        post_fc = self._gated_fc(torch.cat((all_post_emb, all_post_hid), dim=-1))
                        post_feats = post_fc.view(batch_size, -1, dim)
                        combined = torch.cat([pre_feats, compressed_feats, post_feats], dim=1)
                    else:
                        combined = self._compress_variable_length(
                            hidden_states, inputs_embeds, mid_mask, pre_mask, post_mask,
                            mid_lengths, batch_size, dim, n_img
                        )
                else:
                    combined = self._compress_variable_length(
                        hidden_states, inputs_embeds, mid_mask, pre_mask, post_mask,
                        mid_lengths, batch_size, dim, n_img
                    )
        else:
            text_input = torch.cat((inputs_embeds, hidden_states), dim=-1)
            combined = self._gated_fc(text_input)

        return combined, compression_applied, start_indices, mid_lengths


    def forward(
        self,
        hidden_states,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        std=None,
        training_format: Optional[bool] = None,
    ):
        """Forward pass implementing the training pipeline.

        1. Split hidden_states (F) into F_V (visual) and F_T (text)
        2. F_V -> Visual Compressor -> F_hat_V
        3. concat(F_T, E) -> FC -> F_hat_T
        4. concat(F_hat_V, F_hat_T) -> Semi-AR Head (Design A) -> (B, L', k, D)
        5. Training: expand to (B, L, k, D); inference: rearrange to (B, L'*k, D).

        Args:
            training_format: If True, return (B, L, k, D) regardless of self.training.
                If None (default), follow self.training.
        """
        batch_size, seq_length, _ = hidden_states.shape
        use_training_format = training_format if training_format is not None else self.training

        # --- Steps 1-3: compression + FC (delegated to ``compute_post_fc``) ---
        combined, compression_applied, start_indices, mid_lengths = self.compute_post_fc(
            hidden_states, input_ids, inputs_embeds
        )

        # Adjust attention_mask to match compressed sequence length
        padding_mask = attention_mask
        L_combined = combined.shape[1]
        if padding_mask is not None and padding_mask.shape[1] != L_combined:
            if compression_applied and mid_lengths is not None:
                n_images = torch.clamp(mid_lengths // self.image_token_per_image, min=1)
                compress_total = n_images * self.compress_len
                real_orig = padding_mask.sum(dim=1)
                real_new = (real_orig - (mid_lengths - compress_total).clamp(min=0)).clamp(max=L_combined)
                new_mask = torch.zeros(batch_size, L_combined, dtype=padding_mask.dtype, device=padding_mask.device)
                for b in range(batch_size):
                    new_mask[b, :int(real_new[b].item())] = 1
                padding_mask = new_mask
            else:
                padding_mask = padding_mask[:, :L_combined]

        # --- Step 4: SAR head (Design A: block-diagonal placeholder parallel) ---
        layer_outputs = self.layers(combined, padding_mask=padding_mask)  # (B, L', k, D)

        if use_training_format:
            if compression_applied:
                layer_outputs = self._expand_to_original_length(
                    layer_outputs, start_indices, mid_lengths, seq_length
                )
            # Per-slot gated residual: gate[i] controls how much h_t contributes to slot i
            gate = self.residual_gate[None, None, :, None]  # (1, 1, k, 1)
            layer_outputs = layer_outputs + gate * hidden_states.unsqueeze(2)
            return layer_outputs
        else:
            layer_outputs = rearrange_block_diag(layer_outputs, pad_index=self.padding_idx)
            return layer_outputs
    
if __name__ == "__main__":
    batch_size = 2
    seq_len = 2048
    dim = 4096
    k = 4
    
    model = LlamaSemiARHead(
        dim=dim,
        k=k,
        num_heads=64,
        head_dim=64,  # 512 / 8 = 64
        group_size=4
    )
    
    x = torch.randn(batch_size, seq_len, dim)
    output = model(x)
    
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)  # 应该输出 (2, 2048, 5, 512)

