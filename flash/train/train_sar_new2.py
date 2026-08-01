import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
os.environ["WANDB_MODE"] = "offline"
parser = argparse.ArgumentParser(description="sp")
parser.add_argument("--basepath", type=str, default="path/to/llava-1.5-7b-hf")
parser.add_argument(
    "--configpath",
    type=str,
    default="flash/train/nar_config.json",
)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--bs", type=int, default=2)
parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
parser.add_argument(
    "--tmpdir", type=str, default='path/to/train_data'
)
parser.add_argument(
    "--outdir", type=str, default="../weights2"
)
parser.add_argument(
    "--cpdir", type=str, default="../weights2"
)
args = parser.parse_args()

train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": 40,
    # Depending on your data and model size, the larger the model, the higher the sample efficiency. We recommend setting it between 20-40.
    "num_warmup_steps": 1000,
    "total_steps": 100000,
    "p_w": 0.1,
    "v_w": 1.0,
    "head_w": 0.1,
    "num_workers": 10,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 4096,
    # During training, truncating the training sequences means that the larger the setting, the more training data is used, and the better the effect, but it also consumes more VRAM.
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 2,
}
import json
from safetensors import safe_open

# from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
)
from flash.model.llama_sar_new2 import Model
from flash.model.configs import EConfig
from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# import accelerate
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoConfig

if accelerator.is_main_process:
    import wandb

    wandb.init(project="specflash", config=train_config)

baseconfig = AutoConfig.from_pretrained(args.basepath)

image_token_index = baseconfig.image_token_index

head = torch.nn.Linear(
    baseconfig.text_config.hidden_size, baseconfig.vocab_size, bias=False
)

try:
    with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["language_model.lm_head.weight"]
    with safe_open(
        os.path.join(args.basepath, head_path), framework="pt", device="cpu"
    ) as f:
        tensor_slice = f.get_slice("language_model.lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except:
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["language_model.lm_head.weight"]
    weights = torch.load(os.path.join(args.basepath, head_path))
    tensor = weights["language_model.lm_head.weight"].float()

head.weight.data = tensor
head.eval()

for param in head.parameters():
    param.requires_grad = False


def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # try:
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data["hidden_state"][: train_config["max_len"]][None, :]
        inputs_embeds = data["inputs_embeds"][: train_config["max_len"]][None, :]
        input_ids = data["input_ids"][: train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][: train_config["max_len"]][None, :]

        

        length = hidden_state.shape[1]
        # length_q = data['query_ids'].shape[1]
        attention_mask = [1] * length
        # new_loss_mask = new_loss_mask[0].tolist()
        # new_loss_mask[-1] = 0
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        inputs_embeds_target = inputs_embeds[:, 1:, :]
        zeropadding = torch.zeros(1, 1, inputs_embeds_target.shape[2])
        inputs_embeds_target = torch.cat((inputs_embeds_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target
        new_data["inputs_embeds"] = inputs_embeds_target
        # sample = torch.cat((data['xs'],data['xb']))
        # sample=torch.cat((self.data[index]['x'],self.data[index]['logits']))
        # label = data['y']

        if self.transform:
            new_data = self.transform(new_data)

        return new_data



class DataCollatorWithPadding:
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["hidden_state_big"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_inputs_embeds = torch.cat(
            [self.paddingtensor(item["inputs_embeds"], max_length) for item in features]
        )
        batch_hidden_states = torch.cat(
            [
                self.paddingtensor(item["hidden_state_big"], max_length)
                for item in features
            ]
        )
        batch_target = torch.cat(
            [self.paddingtensor(item["target"], max_length) for item in features]
        )
        batch_loss_mask = torch.tensor(
            [
                item["loss_mask"] + [0] * (max_length - len(item["loss_mask"]))
                for item in features
            ]
        )
        batch_attention_mask = torch.tensor(
            [
                item["attention_mask"]
                + [0] * (max_length - len(item["attention_mask"]))
                for item in features
            ]
        )
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids": batch_input_ids,
            "inputs_embeds": batch_inputs_embeds,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch

class CustomDataset_sar(Dataset):
    def __init__(self, datapath, transform=None, max_len=2048, k=5):
        self.data = datapath
        self.transform = transform
        self.max_len = max_len
        self.k = k

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index])
        # Unify shape -> (L, ...)
        hidden_state = data["hidden_state"]
        inputs_embeds = data["inputs_embeds"]
        input_ids = data["input_ids"]
        loss_mask = data["loss_mask"] if "loss_mask" in data else torch.ones(hidden_state.shape[0])

        # Remove batch dim if present
        if hidden_state.ndim == 3 and hidden_state.shape[0] == 1:
            hidden_state = hidden_state[0]
        if inputs_embeds.ndim == 3 and inputs_embeds.shape[0] == 1:
            inputs_embeds = inputs_embeds[0]
        if input_ids.ndim == 2 and input_ids.shape[0] == 1:
            input_ids = input_ids[0]
        if loss_mask.ndim == 2 and loss_mask.shape[0] == 1:
            loss_mask = loss_mask[0]

        # Truncate
        L = min(hidden_state.shape[0], self.max_len)
        hidden_state = hidden_state[:L]
        inputs_embeds = inputs_embeds[:L]
        input_ids = input_ids[:L]
        loss_mask = loss_mask[:L]

        # Shift inputs_embeds by 1: position t gets embed(token_{t+1})
        # This matches EAGLE's convention — the model sees the next token's
        # embedding alongside the current hidden state.
        # input_ids stays unshifted (used for image token detection).
        inputs_embeds_shifted = torch.zeros_like(inputs_embeds)
        inputs_embeds_shifted[:-1] = inputs_embeds[1:]

        inputs_embeds = inputs_embeds_shifted

        k = self.k
        H = hidden_state.shape[-1]

        # Build k-step targets: for position t, target is hidden_state[t+1], ..., hidden_state[t+k]
        # target_hidden: (L, k, H)
        # target_ids: (L, k)
        # target_loss_mask: (L, k)
        target_hidden_list = []
        target_ids_list = []
        target_loss_mask_list = []

        for offset in range(1, k + 1):
            # Shift hidden states by offset positions
            if offset < L:
                shifted_h = torch.cat([hidden_state[offset:], torch.zeros(offset, H)], dim=0)
                shifted_ids = torch.cat([input_ids[offset:], torch.zeros(offset, dtype=input_ids.dtype)], dim=0)
                shifted_mask = torch.cat([loss_mask[offset:], torch.zeros(offset, dtype=loss_mask.dtype)], dim=0)
            else:
                shifted_h = torch.zeros(L, H)
                shifted_ids = torch.zeros(L, dtype=input_ids.dtype)
                shifted_mask = torch.zeros(L, dtype=loss_mask.dtype)
            target_hidden_list.append(shifted_h)
            target_ids_list.append(shifted_ids)
            target_loss_mask_list.append(shifted_mask)

        # Stack: (L, k, H) and (L, k)
        target_hidden = torch.stack(target_hidden_list, dim=1)  # (L, k, H)
        target_ids = torch.stack(target_ids_list, dim=1)  # (L, k)
        target_loss_mask = torch.stack(target_loss_mask_list, dim=1)  # (L, k)

        # Also mask last k positions of the loss mask (incomplete targets)
        target_loss_mask[-k:] = 0

        out = {
            "hidden_state": hidden_state,       # (L, H)
            "inputs_embeds": inputs_embeds,     # (L, H)
            "input_ids": input_ids,             # (L,)
            "target_hidden": target_hidden,     # (L, k, H)
            "target_ids": target_ids,           # (L, k)
            "loss_mask": target_loss_mask,      # (L, k)
        }
        if self.transform:
            # Apply noise transform to hidden_state
            wrapped = {"hidden_state_big": out["hidden_state"][None, :]}
            wrapped = self.transform(wrapped)
            out["hidden_state"] = wrapped["hidden_state_big"][0]
        return out
class DataCollatorSAR:
    """Collator for Semi-AR training: pads variable-length sequences and k-step targets."""

    def paddingtensor(self, tensor, max_len, pad_dim=0):
        """Pad tensor along pad_dim to max_len."""
        pad_size = list(tensor.shape)
        pad_size[pad_dim] = max_len - tensor.shape[pad_dim]
        padding = torch.zeros(*pad_size, dtype=tensor.dtype)
        return torch.cat([tensor, padding], dim=pad_dim)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["hidden_state"].shape[0] for item in features)
        k = features[0]["target_hidden"].shape[1]
        H = features[0]["hidden_state"].shape[-1]

        batch_hidden = []
        batch_embeds = []
        batch_ids = []
        batch_target_hidden = []
        batch_target_ids = []
        batch_loss_mask = []
        batch_attention_mask = []

        for item in features:
            L = item["hidden_state"].shape[0]

            batch_hidden.append(self.paddingtensor(item["hidden_state"], max_length))
            batch_embeds.append(self.paddingtensor(item["inputs_embeds"], max_length))
            batch_ids.append(self.paddingtensor(item["input_ids"], max_length))
            batch_target_hidden.append(self.paddingtensor(item["target_hidden"], max_length))
            batch_target_ids.append(self.paddingtensor(item["target_ids"], max_length))
            batch_loss_mask.append(self.paddingtensor(item["loss_mask"], max_length))
            attn = torch.zeros(max_length, dtype=torch.long)
            attn[:L] = 1
            batch_attention_mask.append(attn)

        batch = {
            "hidden_states": torch.stack(batch_hidden),          # (B, L, H)
            "inputs_embeds": torch.stack(batch_embeds),          # (B, L, H)
            "input_ids": torch.stack(batch_ids),                 # (B, L)
            "target_hidden": torch.stack(batch_target_hidden),   # (B, L, k, H)
            "target_ids": torch.stack(batch_target_ids),         # (B, L, k)
            "loss_mask": torch.stack(batch_loss_mask),           # (B, L, k)
            "attention_mask": torch.stack(batch_attention_mask), # (B, L)
        }
        return batch


def build_nar_samples_from_example(example: Dict, k: int, placeholder_id: int):
    ids_all = example["input_ids"]
    hs_all = example["hidden_state"]
    ie_all = example["inputs_embeds"]
    L = ids_all.shape[0]
    H = hs_all.shape[1]
    D = ie_all.shape[1]

    subs = []
    for t in range(L):
        prefix_len = t + 1
        if prefix_len >= L:
            break  # no next token to predict

        # input ids: prefix + k placeholders
        ph_ids = torch.full((k,), placeholder_id, dtype=ids_all.dtype)
        input_ids_new = torch.cat([ids_all[:prefix_len], ph_ids], dim=0)  # (prefix_len + k,)

        # inputs_embeds: prefix embeds + zeros for placeholders
        pad_ie = torch.zeros((k, D), dtype=ie_all.dtype)
        inputs_embeds_new = torch.cat([ie_all[:prefix_len], pad_ie], dim=0)  # (S, D)

        # hidden_states: prefix hidden + zeros for placeholders (placeholders have no true hidden yet)
        pad_hs = torch.zeros((k, H), dtype=hs_all.dtype)
        hidden_states_new = torch.cat([hs_all[:prefix_len], pad_hs], dim=0)  # (S, H)

        attention_mask_new = torch.ones(input_ids_new.shape[0], dtype=torch.long)

        # targets for k steps
        target_ids = []
        target_hidden = []
        loss_mask = []
        for j in range(1, k + 1):
            pos = t + j
            if pos < L:
                target_ids.append(int(ids_all[pos].item()))
                target_hidden.append(hs_all[pos].unsqueeze(0))
                loss_mask.append(1.0)
            else:
                target_ids.append(-100)
                target_hidden.append(torch.zeros(1, H))
                loss_mask.append(0.0)
        target_ids = torch.tensor(target_ids, dtype=torch.long)         # (k,)
        target_hidden = torch.cat(target_hidden, dim=0)                 # (k, H)
        loss_mask = torch.tensor(loss_mask, dtype=torch.float32)        # (k,)

        subs.append({
            "input_ids": input_ids_new,
            "inputs_embeds": inputs_embeds_new,
            "hidden_states": hidden_states_new,
            "attention_mask": attention_mask_new,
            "target_ids": target_ids,
            "target_hidden": target_hidden,
            "loss_mask": loss_mask,
            "ph_start": prefix_len   # placeholder starts at index = prefix_len (0-based)
        })
    return subs
def top_accuracy(output, target, topk=(1,), mask=None):
    with torch.no_grad():
        if output.dim() == 3:
            B, k, V = output.shape
            out_flat = output.reshape(-1, V)
            tgt_flat = target.reshape(-1)
            if mask is not None:
                mask_flat = mask.reshape(-1).bool()
                if mask_flat.sum() == 0:
                    return [torch.zeros(1, device=out_flat.device) for _ in topk]
                out_flat = out_flat[mask_flat]
                tgt_flat = tgt_flat[mask_flat]
            else:
                valid = tgt_flat != -100
                if valid.sum() == 0:
                    return [torch.zeros(1, device=out_flat.device) for _ in topk]
                out_flat = out_flat[valid]
                tgt_flat = tgt_flat[valid]
        else:
            out_flat = output
            tgt_flat = target

        if tgt_flat.numel() == 0:
            return [torch.zeros(1, device=out_flat.device) for _ in topk]

        maxk = max(topk)
        _, pred = out_flat.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(tgt_flat.view(1, -1).expand_as(pred))

        res = []
        for k_ in topk:
            correct_k = correct[:k_].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


@torch.no_grad()
def getkacc(model, data, head, max_length=5):
    hidden_states = data["hidden_states"]
    input_ids = data["input_ids"]
    inputs_embeds = data["inputs_embeds"]
    # attention_mask=data["attention_mask"]
    loss_mask = data["loss_mask"]
    # sample_mask=data["sample_mask"]
    target = data["target"]
    total = [0 for _ in range(max_length)]
    correct = [0 for _ in range(max_length)]
    bs, sl = hidden_states.shape[0], hidden_states.shape[1]
    target_headout = head(target)
    hidden_states_headout = head(hidden_states)

    for i in range(bs):
        for j in range(sl):

            single_hidden_states = hidden_states[i, :j]
            single_input_ids = input_ids[i, :j]
            single_inputs_embeds = inputs_embeds[i, :j]

            single_hidden_states = single_hidden_states[None, :, :]
            single_input_ids = single_input_ids[None, :]
            single_inputs_embeds = single_inputs_embeds[None, :, :]
            for k in range(max_length):
                if loss_mask[i, single_hidden_states.shape[1] - 1] == 0:
                    break
                tmp_in_target_headout = hidden_states_headout[
                    i, single_hidden_states.shape[1] - 1
                ]
                tmp_out_target_headout = target_headout[
                    i, single_hidden_states.shape[1] - 1
                ]
                target_in_token = torch.argmax(tmp_in_target_headout)
                target_out_token = torch.argmax(tmp_out_target_headout)
                tmp_token = input_ids[i, single_hidden_states.shape[1] - 1]
                # tmp_sample_mask=sample_mask[i,single_hidden_states.shape[1]-1]
                if not (target_in_token == tmp_token):
                    break
                out_hidden = model(
                    single_hidden_states,
                    input_ids=single_input_ids,
                    inputs_embeds=single_inputs_embeds,
                )
                last_hidden = out_hidden[:, -1]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout)
                total[k] += 1
                if token == target_out_token:
                    correct[k] += 1
                else:
                    for kk in range(k + 1, max_length):
                        total[kk] += 1
                    break

                single_hidden_states = torch.cat(
                    (single_hidden_states, out_hidden[:, -1:]), dim=1
                )
                single_input_ids = torch.cat(
                    (
                        single_input_ids,
                        torch.tensor([[token]]).to(single_input_ids.device),
                    ),
                    dim=1,
                )
                single_inputs_embeds = torch.cat(
                    (
                        single_inputs_embeds,
                        model.embed_tokens(token.unsqueeze(0)).unsqueeze(0),
                    ),
                    dim=1,
                )

    acc = [correct[i] / total[i] for i in range(len(correct))]
    return acc


if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    aug = None

import random
datapath = list_files(train_config["datapath"])
random.seed(42)
random.shuffle(datapath)

traindatapath = datapath[: int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95) :]

config = EConfig.from_pretrained(train_config["config_path"])
k_steps = int(config.k)

traindataset = CustomDataset_sar(traindatapath, transform=aug, k=k_steps)
testdataset = CustomDataset_sar(testdatapath, k=k_steps)
train_loader = DataLoader(
    traindataset,
    batch_size=train_config["bs"],
    shuffle=True,
    collate_fn=DataCollatorSAR(),
    num_workers=train_config["num_workers"],
    pin_memory=True,
)
test_loader = DataLoader(
    testdataset,
    batch_size=train_config["bs"],
    shuffle=False,
    collate_fn=DataCollatorSAR(),
    num_workers=train_config["num_workers"],
    pin_memory=True,
)
# for batch_data in train_loader:
#     print(batch_data)

if accelerator.is_main_process:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

# config already loaded above (before dataset creation)
model = Model(config, load_emb=True, path=args.basepath)

criterion = nn.SmoothL1Loss(reduction="none")
optimizer = optim.AdamW(
    model.parameters(),
    lr=train_config["lr"],
    betas=(train_config["b1"], train_config["b2"]),
    weight_decay=0.05,
)

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

if is_warmup:
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )
else:
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )
# Move frozen head to device without FSDP wrapping (it's inference-only, no gradients)
head = head.to(accelerator.device)
# accelerator.load_state("checkpoints/state_5")
for epoch in range(num_epochs + 1):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    # Per-slot accuracy tracking for training
    train_correct_per_slot = [0 for _ in range(k_steps)]
    train_total_per_slot = [0 for _ in range(k_steps)]

    # --- Scheduled Sampling: linearly increase ss_prob from 0 to 0.5 over training ---
    # First 2 epochs: pure teacher forcing (ss_prob=0) for stable warmup.
    # Then linearly ramp to ss_max over remaining epochs.
    ss_max = 0.2
    ss_warmup_epochs = 4
    if epoch < ss_warmup_epochs:
        ss_prob = 0.0
    else:
        ss_prob = min(ss_max, ss_max * (epoch - ss_warmup_epochs) / max(num_epochs - ss_warmup_epochs, 1))
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.ss_prob = ss_prob
    if accelerator.is_local_main_process:
        print(f"Epoch {epoch+1}: scheduled_sampling_prob = {ss_prob:.3f}")

    model.train()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        with accelerator.accumulate(model):
            optimizer.zero_grad()

            predict = model(
                data["hidden_states"],
                input_ids=data["input_ids"],
                inputs_embeds=data["inputs_embeds"],
                attention_mask=data.get("attention_mask"),
                target_ids=data["target_ids"],
                lm_head=head,
            )

            # predict shape: (B, L_pred, k, D) — direct output from Semi-AR Head.
            # During training the model uses FC-only (no compression), so L_pred == L.
            B, L_pred, k, D = predict.shape

            # Align targets with model output length.
            # Since L_pred == L during training, [:, :L_pred] is effectively a no-op.
            target_hidden = data["target_hidden"][:, :L_pred, :, :]  # (B, L_pred, k, H)
            loss_mask = data["loss_mask"][:, :L_pred, :]  # (B, L_pred, k)
            target_ids = data["target_ids"][:, :L_pred, :]  # (B, L_pred, k)

            predict_reshaped = predict  # already (B, L_pred, k, D)

            # --- Regression loss (on hidden states) ---
            # predict_reshaped: (B, L_pred, k, D), target_hidden: (B, L_pred, k, H)
            vloss = criterion(predict_reshaped, target_hidden)
            # vloss: (B, L_pred, k, D) -> mean over D, mask over (L_pred, k)
            loss_mask_expanded = loss_mask.unsqueeze(-1)  # (B, L_pred, k, 1)
            vloss = torch.sum(torch.mean(loss_mask_expanded * vloss, dim=-1)) / (
                loss_mask.sum() + 1e-5
            )

            # --- Classification loss (on token distributions) ---
            with torch.no_grad():
                # Target distributions from frozen LM head
                target_logits = head(target_hidden)  # (B, L_pred, k, V)
                target_p = torch.softmax(target_logits, dim=-1).detach()

            # Predicted distributions
            out_logits = head(predict_reshaped)  # (B, L_pred, k, V)
            out_logp = torch.log_softmax(out_logits, dim=-1)

            # KL-divergence style loss: -sum(target_p * log(pred_p))
            plogp = target_p * out_logp  # (B, L_pred, k, V)
            # Sum over vocab dim, average over valid positions
            ploss = -torch.sum(torch.sum(loss_mask_expanded * plogp, dim=-1)) / (
                loss_mask.sum() + 1e-5
            )

            loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
            accelerator.backward(loss)
            accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            if is_warmup:
                scheduler.step()

        # --- Logging metrics ---
        with torch.no_grad():
            _, predicted = torch.max(out_logits, dim=-1)  # (B, L_pred, k)
            _, target_tokens = torch.max(target_logits, dim=-1)  # (B, L_pred, k)
            ct = loss_mask.sum().item()
            cc = ((predicted == target_tokens) * loss_mask).sum().item()
            total += ct
            correct += cc
            # Per-slot accumulation
            for ki in range(k_steps):
                slot_mask_ki = loss_mask[:, :, ki]
                train_correct_per_slot[ki] += ((predicted[:, :, ki] == target_tokens[:, :, ki]) * slot_mask_ki).sum().item()
                train_total_per_slot[ki] += slot_mask_ki.sum().item()

        if accelerator.is_main_process and ct != 0:
            logdict = {
                "train/lr": optimizer.optimizer.param_groups[0]["lr"],
                "train/vloss": vloss.item(),
                "train/ploss": ploss.item(),
                "train/loss": loss.item(),
                "train/acc": cc / ct,
            }
            wandb.log(logdict)
            if batch_idx % 2000 == 0:
                print(f"  [step {batch_idx}] vloss={vloss.item():.4f} ploss={ploss.item():.4f} "
                      f"weighted: v={train_config['v_w']*vloss.item():.4f} p={train_config['p_w']*ploss.item():.4f} "
                      f"acc={cc/ct:.4f}")

        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1

    # --- Epoch-level logging ---
    correct_t, total_t = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct_t, total_t = accelerator.gather_for_metrics((correct_t, total_t))
    correct_sum, total_sum = correct_t.sum().item(), total_t.sum().item()
    epoch_loss /= max(num_batches, 1)

    # Gather per-slot training metrics
    train_correct_per_slot_t = torch.tensor(train_correct_per_slot, dtype=torch.float64).cuda()
    train_total_per_slot_t = torch.tensor(train_total_per_slot, dtype=torch.float64).cuda()
    train_correct_per_slot_t, train_total_per_slot_t = accelerator.gather_for_metrics(
        (train_correct_per_slot_t, train_total_per_slot_t)
    )
    if train_correct_per_slot_t.dim() > 1:
        train_correct_per_slot_sum = train_correct_per_slot_t.sum(dim=0)
        train_total_per_slot_sum = train_total_per_slot_t.sum(dim=0)
    else:
        train_correct_per_slot_sum = train_correct_per_slot_t
        train_total_per_slot_sum = train_total_per_slot_t

    if accelerator.is_local_main_process:
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, epoch_loss))
        if total_sum > 0:
            print("Train Accuracy: {:.2f}%".format(100 * correct_sum / total_sum))
        # Print per-slot training accuracy
        train_slot_acc_dict = {}
        for ki in range(k_steps):
            slot_total_ki = train_total_per_slot_sum[ki].item()
            slot_correct_ki = train_correct_per_slot_sum[ki].item()
            slot_acc = 100 * slot_correct_ki / max(slot_total_ki, 1)
            print(f"  Train Slot {ki} (predict t+{ki+1}): {slot_acc:.2f}%")
            train_slot_acc_dict[f"train/slot{ki}_acc"] = slot_correct_ki / max(slot_total_ki, 1)
        wandb.log({"train/epochacc": correct_sum / max(total_sum, 1), "train/epochloss": epoch_loss, "train/ss_prob": ss_prob, **train_slot_acc_dict})

    # --- Save checkpoint ---
    if accelerator.is_local_main_process and (epoch + 1) % train_config["save_freq"] == 0:
        save_path = os.path.join(args.cpdir, f"state_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        # FSDP-compatible save: unwrap model to get raw state_dict
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        config.save_pretrained(save_path)
        print(f"Checkpoint saved to {save_path}")

    # --- Evaluation ---
    if (epoch + 1) % train_config["save_freq"] == 0:
        model.eval()
        eval_loss = 0
        eval_correct = 0
        eval_total = 0
        eval_batches = 0
        # Per-slot accuracy tracking
        eval_correct_per_slot = [0 for _ in range(k_steps)]
        eval_total_per_slot = [0 for _ in range(k_steps)]

        for batch_idx, data in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                predict = model(
                    data["hidden_states"],
                    input_ids=data["input_ids"],
                    inputs_embeds=data["inputs_embeds"],
                    attention_mask=data.get("attention_mask"),
                    training_format=True,
                    target_ids=data["target_ids"],
                    lm_head=head,
                )

                B, L_pred, k, D = predict.shape
                predict_reshaped = predict  # already (B, L_pred, k, D)

                target_hidden = data["target_hidden"][:, :L_pred, :, :]
                loss_mask_eval = data["loss_mask"][:, :L_pred, :]
                loss_mask_expanded_eval = loss_mask_eval.unsqueeze(-1)

                # Regression loss
                vloss_eval = criterion(predict_reshaped, target_hidden)
                vloss_eval = torch.sum(torch.mean(loss_mask_expanded_eval * vloss_eval, dim=-1)) / (
                    loss_mask_eval.sum() + 1e-5
                )

                # Classification loss
                target_logits_eval = head(target_hidden)
                target_p_eval = torch.softmax(target_logits_eval, dim=-1)
                out_logits_eval = head(predict_reshaped)
                out_logp_eval = torch.log_softmax(out_logits_eval, dim=-1)
                ploss_eval = -torch.sum(torch.sum(loss_mask_expanded_eval * target_p_eval * out_logp_eval, dim=-1)) / (
                    loss_mask_eval.sum() + 1e-5
                )

                loss_eval = train_config["v_w"] * vloss_eval + train_config["p_w"] * ploss_eval

                _, pred_tokens = torch.max(out_logits_eval, dim=-1)
                _, tgt_tokens = torch.max(target_logits_eval, dim=-1)
                ct_eval = loss_mask_eval.sum().item()
                cc_eval = ((pred_tokens == tgt_tokens) * loss_mask_eval).sum().item()
                eval_correct += cc_eval
                eval_total += ct_eval
                eval_loss += loss_eval.item()
                eval_batches += 1

                # Per-slot accuracy accumulation
                for ki in range(k_steps):
                    slot_mask = loss_mask_eval[:, :, ki]
                    slot_correct = ((pred_tokens[:, :, ki] == tgt_tokens[:, :, ki]) * slot_mask).sum().item()
                    slot_total = slot_mask.sum().item()
                    eval_correct_per_slot[ki] += slot_correct
                    eval_total_per_slot[ki] += slot_total

        eval_correct_t = torch.tensor(eval_correct).cuda()
        eval_total_t = torch.tensor(eval_total).cuda()
        eval_correct_t, eval_total_t = accelerator.gather_for_metrics((eval_correct_t, eval_total_t))
        eval_correct_sum = eval_correct_t.sum().item()
        eval_total_sum = eval_total_t.sum().item()
        eval_loss /= max(eval_batches, 1)

        # Gather per-slot metrics across processes
        eval_correct_per_slot_t = torch.tensor(eval_correct_per_slot, dtype=torch.float64).cuda()
        eval_total_per_slot_t = torch.tensor(eval_total_per_slot, dtype=torch.float64).cuda()
        eval_correct_per_slot_t, eval_total_per_slot_t = accelerator.gather_for_metrics(
            (eval_correct_per_slot_t, eval_total_per_slot_t)
        )
        # After gather: shape may be (num_processes, k_steps), sum over dim 0
        if eval_correct_per_slot_t.dim() > 1:
            eval_correct_per_slot_sum = eval_correct_per_slot_t.sum(dim=0)
            eval_total_per_slot_sum = eval_total_per_slot_t.sum(dim=0)
        else:
            eval_correct_per_slot_sum = eval_correct_per_slot_t
            eval_total_per_slot_sum = eval_total_per_slot_t

        if accelerator.is_local_main_process:
            print("Test Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, eval_loss))
            if eval_total_sum > 0:
                print("Test Accuracy: {:.2f}%".format(100 * eval_correct_sum / eval_total_sum))
            # Print per-slot accuracy
            slot_acc_dict = {}
            for ki in range(k_steps):
                slot_total_ki = eval_total_per_slot_sum[ki].item()
                slot_correct_ki = eval_correct_per_slot_sum[ki].item()
                slot_acc = 100 * slot_correct_ki / max(slot_total_ki, 1)
                print(f"  Slot {ki} (predict t+{ki+1}): {slot_acc:.2f}%  ({int(slot_correct_ki)}/{int(slot_total_ki)})")
                slot_acc_dict[f"test/slot{ki}_acc"] = slot_correct_ki / max(slot_total_ki, 1)
            wandb.log({
                "test/epochacc": eval_correct_sum / max(eval_total_sum, 1),
                "test/epochloss": eval_loss,
                **slot_acc_dict,
            })
