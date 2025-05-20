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

    def forward(self, q, k):
        # q形状: [batch, num_heads, seq_len, head_dim]
        # k形状: [batch, group_size, seq_len, head_dim]
        seq_len = q.size(2)
        device = q.device

        # 生成位置编码
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]

        # 扩展维度用于广播
        cos = emb.cos()[None, None, :, None, :]  # [1, 1, seq_len, 1, dim]
        sin = emb.sin()[None, None, :, None, :]

        # 应用旋转到每个头
        q_rot = self._apply_rotary(q.unsqueeze(-2), cos, sin).squeeze(-2)
        k_rot = self._apply_rotary(k.unsqueeze(-2), cos, sin).squeeze(-2)
        
        return q_rot, k_rot

class LlamaSemiARHead(nn.Module):
    def __init__(self, dim, k, num_heads=8, head_dim=128, group_size=4):
        super().__init__()
        # 添加维度验证
        assert dim == num_heads * head_dim, f"dim({dim})必须等于num_heads×head_dim({num_heads}×{head_dim})"
        assert head_dim % 2 == 0, "head_dim必须是偶数"
        assert group_size <= num_heads, "group_size不能超过num_heads"

        self.dim = dim
        self.k = k
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.group_size = group_size

        # 投影层
        self.q_proj = nn.Linear(dim, num_heads * head_dim)
        self.k_proj = nn.Linear(dim, group_size * head_dim)
        self.v_proj = nn.Linear(dim, group_size * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, dim)

        # 旋转位置编码
        self.rotary_emb = RotaryEmbedding(head_dim)

        # 并行预测头
        self.ar_proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, k * dim)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 生成Q/K/V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(batch_size, seq_len, self.group_size, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(batch_size, seq_len, self.group_size, self.head_dim).permute(0, 2, 1, 3)

        # 应用旋转编码
        q, k = self.rotary_emb(q, k)

        # 注意力计算
        attn_scores = torch.einsum('bhid,bgjd->bhgij', q, k) / (self.head_dim ** 0.5)
        
        # 因果掩码
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask[None, None, None, :, :], float('-inf'))
        
        # 注意力输出
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.einsum('bhgij,bgjd->bhid', attn_weights, v)
        
        # 合并多头输出
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        # 生成k个预测
        output = self.ar_proj(attn_output).view(batch_size, seq_len, self.k, self.dim)
        
        return output

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Model(nn.Module):
    def __init__(self, config, load_emb=False, path=None, bias=True):
        super().__init__()

        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        self.layers = LlamaSemiARHead(
        dim=config.hidden_size,
        k=config.k,
        num_heads=config.num_attention_heads,
        head_dim=config.hidden_size//config.num_attention_heads,  # 512 / 8 = 64
        group_size=4
    
        )
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias)

    


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
    ):
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # with torch.no_grad():
        #     inputs_embeds = self.embed_tokens(input_ids)
        # inputs_embeds = inputs_embeds.detach()

        # if std is not None:
        #     noise = torch.randn(inputs_embeds.size(),device=inputs_embeds.device) * std
        #     inputs_embeds=inputs_embeds+noise
        if inputs_embeds is None:

            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = (
                hidden_states.device
                if hidden_states is not None
                else inputs_embeds.device
            )
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device,
            )
   

        # if self.gradient_checkpointing and self.training:
        #    if use_cache:
        #        use_cache = False

        # hidden_states=self.act(self.fc(torch.cat((inputs_embeds,hidden_states),dim=-1)))
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        #print(inputs_embeds.shape,hidden_states.shape)
        if compress_flag is None:
            hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))
        else:
            mask = (input_ids == 32000)
            start_indices = mask.int().argmax(dim=1)
            mid_lengths = mask.sum(dim=1)
            pos = torch.arange(seq_length, device=hidden_states.device).expand_as(hidden_states[..., 0])

            # mask
            pre_mask = pos < start_indices.unsqueeze(-1)
            mid_mask = (pos >= start_indices.unsqueeze(-1)) & (pos < (start_indices + mid_lengths).unsqueeze(-1))
            post_mask = pos >= (start_indices + mid_lengths).unsqueeze(-1)

            # sep
            output = torch.zeros_like(hidden_states).half()
            output[pre_mask] = hidden_states[pre_mask].half()

            post_data_emb = inputs_embeds[post_mask]
            post_data = hidden_states[post_mask]
            text_embs = self.fc(torch.cat((post_data_emb, post_data), dim=-1))
            output[post_mask] = text_embs

            # mid
            img_feats = hidden_states[mid_mask].view(batch_size, -1, dim)
            attn_scores = torch.einsum("h d, b t d -> b h t", self.A, img_feats)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            compressed_feats = torch.einsum("b h t, b t d -> b h d", attn_weights, img_feats).view(-1, dim)

            new_mid_lengths = self.A.size(0)
            mid_mask_new = (pos >= start_indices.unsqueeze(-1)) & (
                        pos < (start_indices + new_mid_lengths).unsqueeze(-1))
            output[mid_mask_new] = compressed_feats.half()

            hidden_states = output[:, :64 - 576, :]

        layer_outputs = self.layers(hidden_states)
        layer_outputs = rearrange_block_diag(layer_outputs,pad_index=self.padding_idx)
        return layer_outputs
    
if __name__ == "__main__":
    batch_size = 2
    seq_len = 2048
    dim = 512
    k = 5
    
    model = LlamaSemiARHead(
        dim=dim,
        k=k,
        num_heads=8,
        head_dim=64,  # 512 / 8 = 64
        group_size=4
    )
    
    x = torch.randn(batch_size, seq_len, dim)
    output = model(x)
    
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)  # 应该输出 (2, 2048, 5, 512)

