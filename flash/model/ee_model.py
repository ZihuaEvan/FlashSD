import copy
import json
import time

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig
from .modeling_llava import LlavaForConditionalGeneration
from .modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from .utils import *
from .kv_cache import initialize_past_key_values
from .choices import mc_sim_7b_63
from transformers import AutoTokenizer, AutoProcessor
import os
from huggingface_hub import hf_hub_download
from .cnets import Model
from .llama_sar import Model as SARModel
from .configs import EConfig
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

class EeModel(nn.Module):
    def __init__(
        self,
        base_model,
        base_model_name_or_path,
        ee_model_path,
    ):

        super().__init__()
        self.base_model = base_model
        if type(base_model)==LlavaForConditionalGeneration :
            self.language_model=base_model.language_model.model
            self.lm_head=base_model.language_model.lm_head
        else :
            self.language_model=base_model.model
            self.lm_head=base_model.lm_head
            

        self.config = base_model.config
        self.hidden_size = self.lm_head.weight.shape[-1]
        self.vocab_size = self.lm_head.weight.shape[0]
        
        self.base_model_name_or_path = base_model_name_or_path

        self.processor = AutoProcessor.from_pretrained(self.base_model_name_or_path)

        if type(base_model)==LlavaForConditionalGeneration :
            self.processor.patch_size=self.config.vision_config.patch_size

        config = EConfig.from_pretrained(ee_model_path)
        with open(ee_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        self.ee_layer = Model(config, bias=bias)

        low_memory = False

        device = self.language_model.layers[-1].self_attn.q_proj.weight.device
        if device != self.lm_head.weight.device:
            self.ee_layer.diff_device = True
            if not low_memory:
                self.ee_layer.headweight = (
                    self.lm_head.weight.clone().to(device)
                )
            else:
                self.ee_layer.layer_device = device

        else:
            self.ee_layer.diff_device = False
        self.ee_layer.to(self.base_model.dtype).to(device)
        self.ee_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.processor.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        Type="LLaVA",
        base_model_path=None,
        ee_model_path=None,
        **kwargs,
    ):
        # assert Type=="Llava"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type == "LlavaForConditionalGeneration":
            base_model = LlavaForConditionalGeneration.from_pretrained( base_model_path,**kwargs)

        elif Type == "Qwen2_5_VLForConditionalGeneration":
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model_path,**kwargs)

        configpath = os.path.join(ee_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ee_model_path, "config.json")
        model = cls(base_model, base_model_path, configpath)


        load_model_path = os.path.join(ee_model_path, "model.safetensors")
        if not os.path.exists(load_model_path):
            load_model_path = hf_hub_download(ee_model_path, "model.safetensors")

        ee_layer_state_dict = load_file(load_model_path, device="cuda")


        # load_model_path = os.path.join(ee_model_path, "pytorch_model.bin")
        # if not os.path.exists(load_model_path):
        #     load_model_path = hf_hub_download(ee_model_path, "pytorch_model.bin")
            
        # ee_layer_state_dict = torch.load(
        #     load_model_path, map_location=base_model.device
        # )
        ####
        model.ee_layer.load_state_dict(ee_layer_state_dict, strict=False)

        return model

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        init=True,
        logits_processor=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                output_hidden_states=True,
            )
            hidden_states = outputs["hidden_states"][-1]
            inputs_embeds = outputs["hidden_states"][0]
            if output_orig:
                orig = self.lm_head(hidden_states)
        if init:
            if logits_processor is not None:
                logits = orig[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                token = torch.multinomial(probabilities, 1)
            else:
                token = torch.argmax(orig[:, -1])
                token = token[None, None]
            input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

            inputs_embeds = torch.cat((inputs_embeds, self.ee_layer.embed_tokens(token).to(inputs_embeds.device)), dim=1)


            ea_logits = self.ee_layer.topK_genrate(
                hidden_states=hidden_states,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                model=self.base_model,
                head=self.lm_head,
                logits_processor=logits_processor,
            )
            if output_orig:
                return ea_logits, outputs, orig, hidden_states, token
            return ea_logits, hidden_states, token
        else:
            if output_orig:
                return outputs, orig, hidden_states
    @torch.no_grad()
    def eagenerate(
        self,
        input_ids,
        pixel_values=None,
        image_grid_thw=None,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=4096,
        tree_choices=mc_sim_7b_63,
        **kwargs,
    ):
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.ee_layer.reset_kv()

        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices,
                device=self.language_model.layers[-1].self_attn.q_proj.weight.device,
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.lm_head.weight.device
            )
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.language_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self.language_model)

        tree_logits, logits, hidden_state, sample_token = initialize_tree(
            input_ids=input_ids,
            pixel_values=pixel_values,
            model=self,
            tree_attn_mask=tree_buffers["tree_attn_mask"],
            past_key_values=past_key_values,
            logits_processor=logits_processor,
            image_grid_thw=image_grid_thw,
        )
        new_token = 0

        for idx in range(max_length):
            candidates, cart_candidates_prob, tree_candidates = generate_candidates(
                tree_logits=tree_logits,
                tree_indices=tree_buffers["tree_indices"],
                retrieve_indices=tree_buffers["retrieve_indices"],
                sample_token=sample_token,
                logits_processor=logits_processor,
            )
            logits, hidden_state_new, outputs = tree_decoding(
                model=self,
                tree_candidates=tree_candidates,
                past_key_values=past_key_values,
                tree_position_ids=tree_buffers["tree_position_ids"],
                input_ids=input_ids,
                retrieve_indices=tree_buffers["retrieve_indices_head"],
            )
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits=logits,
                candidates=candidates,
                logits_processor=logits_processor,
                cart_candidates_prob=cart_candidates_prob,
                op=tree_logits[2],
                p_indices=tree_buffers["p_indices"],
                tree_candidates=tree_candidates,
                b_indices=tree_buffers["b_indices"],
            )
            input_ids,tree_logits,new_token,hidden_state,sample_token = update_inference_inputs(
                input_ids=input_ids,
                candidates=candidates,
                best_candidate=best_candidate,
                accept_length=accept_length,
                retrieve_indices=tree_buffers["retrieve_indices"],
                logits_processor=logits_processor,
                logits=logits,
                tree_logits=tree_logits,
                new_token=new_token,
                past_key_values_data_list=past_key_values_data,
                current_length_data=current_length_data,
                model=self,
                hidden_state=hidden_state,
                hidden_state_new=hidden_state_new,
                sample_p=sample_p,
            )

            if self.processor.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                return input_ids
            if new_token > max_new_tokens:
                return input_ids
            if input_ids.shape[1] > max_length:
                return input_ids
        

    @staticmethod
    def _trim_kv_cache(past_key_values, target_len):
        """Trim KV cache to target_len along the sequence dimension."""
        return tuple(
            (k[:, :, :target_len, :], v[:, :, :target_len, :])
            for k, v in past_key_values
        )

    @torch.no_grad()
    def sar_generate(
        self,
        input_ids,
        pixel_values=None,
        image_grid_thw=None,
        temperature=0.0,
        top_p=0.0,
        top_k=0,
        max_new_tokens=512,
        max_length=4096,
        sar_model=None,
        **kwargs,
    ):
        if sar_model is None:
            if not isinstance(self.ee_layer, SARModel):
                raise ValueError(
                    "sar_generate() requires a SARModel instance. "
                    "Pass sar_model=... explicitly."
                )
            sar_model = self.ee_layer

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(
                temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            logits_processor = None

        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!"
        input_ids = input_ids.clone()

        draft_model = sar_model
        k = getattr(draft_model, 'k', 5)
        input_len = input_ids.shape[1]
        new_token = 0

        # ==================== PREFILL ====================
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=True,
            output_hidden_states=True,
        )
        base_kv = outputs.past_key_values
        hidden_all = outputs.hidden_states[-1]   # (1, L, D)
        embeds_all = outputs.hidden_states[0]    # (1, L, D)

        post_fc_buffer, _, _, _ = draft_model.compute_post_fc(
            hidden_all, input_ids, embeds_all
        )

        # Track raw hidden states for residual skip connection
        hidden_buffer = hidden_all

        prev_logit = self.lm_head(hidden_all[:, -1:, :])  # (1, 1, V)

        # ==================== DECODE LOOP ====================
        while True:
            cur_len = input_ids.shape[1]

            # --- 1. SAR draft: predict k tokens from the buffer ---
            sar_out = draft_model.sar_only(
                post_fc_buffer, hidden_states_residual=hidden_buffer
            )  # (1, L', k, D)
            draft_hidden = sar_out[:, -1, :, :]              # (1, k, D)
            draft_logits = self.lm_head(draft_hidden)        # (1, k, V)

            if logits_processor is not None:
                draft_tokens = []
                for i in range(k):
                    logits_i = logits_processor(None, draft_logits[:, i, :])
                    probs_i = torch.softmax(logits_i, dim=-1)
                    token_i = torch.multinomial(probs_i, 1)
                    draft_tokens.append(token_i)
                draft_tokens = torch.cat(draft_tokens, dim=1)  # (1, k)
            else:
                draft_tokens = torch.argmax(draft_logits, dim=-1)  # (1, k)

            # --- 2. Target verifies k draft tokens with KV cache ---
            kv_len_before = base_kv[0][0].shape[2]
            v_out = self.base_model(
                input_ids=draft_tokens,
                past_key_values=base_kv,
                use_cache=True,
                output_hidden_states=True,
            )
            v_kv = v_out.past_key_values
            v_hidden = v_out.hidden_states[-1]  # (1, k, D)
            v_embeds = v_out.hidden_states[0]   # (1, k, D)
            v_logits = self.lm_head(v_hidden)   # (1, k, V)

            # prev_logit verifies draft_tokens[0]
            # v_logits[:, i] verifies draft_tokens[i+1] (and v_logits[:, k-1] is bonus)
            all_logits = torch.cat([prev_logit, v_logits], dim=1)  # (1, k+1, V)

            # --- 3. Accept / reject ---
            if logits_processor is None:
                target_tokens = torch.argmax(all_logits[:, :k, :], dim=-1)
                matches = (draft_tokens == target_tokens)
                accept_mask = torch.cumprod(matches.int(), dim=1)
                accept_length = int(accept_mask.sum(dim=1).item())

                if accept_length < k:
                    extra_token = torch.argmax(
                        all_logits[:, accept_length, :], dim=-1, keepdim=True
                    )
                else:
                    extra_token = torch.argmax(
                        all_logits[:, k, :], dim=-1, keepdim=True
                    )
            else:
                accept_length = 0
                extra_token = None
                for i in range(k):
                    t_logits_i = logits_processor(None, all_logits[:, i, :])
                    t_probs_i = torch.softmax(t_logits_i, dim=-1)
                    d_logits_i = logits_processor(None, draft_logits[:, i, :])
                    d_probs_i = torch.softmax(d_logits_i, dim=-1)

                    tok_i = draft_tokens[0, i].item()
                    p = t_probs_i[0, tok_i].item()
                    q = d_probs_i[0, tok_i].item()

                    if torch.rand(1, device=draft_tokens.device).item() < min(1.0, p / (q + 1e-10)):
                        accept_length += 1
                    else:
                        residual = torch.clamp(t_probs_i - d_probs_i, min=0)
                        residual = residual / (residual.sum() + 1e-10)
                        extra_token = torch.multinomial(residual, 1)
                        break

                if extra_token is None:
                    bonus_logits = logits_processor(None, all_logits[:, k, :])
                    bonus_probs = torch.softmax(bonus_logits, dim=-1)
                    extra_token = torch.multinomial(bonus_probs, 1)

            accepted_ids = torch.cat(
                [draft_tokens[:, :accept_length], extra_token], dim=1
            )  # (1, accept_length + 1)
            n_new = accepted_ids.shape[1]
            input_ids = torch.cat([input_ids, accepted_ids], dim=1)
            new_token += n_new

            # --- 4. Trim target KV cache ---
            # v_kv has kv_len_before + k entries; keep only kv_len_before + accept_length
            base_kv = self._trim_kv_cache(v_kv, kv_len_before + accept_length)

            # Process the extra token (correction / bonus) through target for its KV + hidden
            extra_out = self.base_model(
                input_ids=extra_token,
                past_key_values=base_kv,
                use_cache=True,
                output_hidden_states=True,
            )
            base_kv = extra_out.past_key_values
            extra_hidden = extra_out.hidden_states[-1]  # (1, 1, D)
            extra_embeds = extra_out.hidden_states[0]   # (1, 1, D)

            # --- 5. Update SAR post-FC buffer and hidden buffer ---
            new_hidden = torch.cat([v_hidden[:, :accept_length], extra_hidden], dim=1)
            new_embeds = torch.cat([v_embeds[:, :accept_length], extra_embeds], dim=1)
            new_fc = draft_model.encode_text(new_hidden, new_embeds)
            post_fc_buffer = torch.cat([post_fc_buffer, new_fc], dim=1)
            hidden_buffer = torch.cat([hidden_buffer, new_hidden], dim=1)

            # Next iteration's prev_logit
            prev_logit = self.lm_head(extra_hidden)  # (1, 1, V)

            # --- 6. Stopping conditions ---
            tail = input_ids[0, cur_len:].tolist()
            if self.processor.tokenizer.eos_token_id in tail:
                return input_ids
            if new_token >= max_new_tokens:
                return input_ids
            if input_ids.shape[1] >= max_length:
                return input_ids
    