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
        

        
    