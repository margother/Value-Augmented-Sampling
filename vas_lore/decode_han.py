"""Decoding-side loader for the per-basis value models trained by train_value_han.py (Han et al.'s VASTrainer).

The VAS decoding RULE (top-k candidates of the policy, values zero-filled and centred over the top-k, + beta * value added to the logits, then sample) is the one of
../vas_trainer.py's VASLogitsProcessor. That class itself cannot run in the dpo_LRF environments: it needs the authors' TRL fork (`return_past_key_values`, tuple KV caches),
while transformers 4.5x uses Cache objects. The loop that implements the same rule on Cache objects is dpo_LRF/vas/vas_decoding.py (`vas_generate`); it only needs a
`value_model(input_ids, attention_mask, position_ids, past_key_values, use_cache) -> (values (B, T, K), cache)`. `ValueBank` below is that callable for K INDEPENDENT scalar value
models (the published multi-reward recipe: one value function per reward, combined at inference as sum_k w_k V_k -- the weighted sum is done by vas_generate's `w_eff`).

Values are in units of (basis reward / reward_scale) -- the training target of train_value_han.py -- i.e. LoRe's logit units, so vas_generate is called with sigma = 1 and beta = 1
is the KL-regularised optimum for the user's LoRe reward.

Needs only torch / transformers / peft (no TRL): each basis directory (written by VASTrainer.save_pretrained) holds the LoRA adapter (adapter_config.json,
adapter_model.safetensors) and the value head (pytorch_model.bin, keys v_head.summary.weight / .bias). K models are held merged in memory, ~2.5 GB each for the 1B backbone.
"""
import json
import os
from typing import List, Optional

import torch


class ValueBankCache:
    """K per-model KV caches that look like ONE cache to vas_generate (it deep-copies the cache and calls batch_repeat_interleave)."""

    def __init__(self, caches: List):
        self.caches = caches

    def batch_repeat_interleave(self, repeats: int):
        for c in self.caches:
            c.batch_repeat_interleave(repeats)


class ValueBank(torch.nn.Module):
    def __init__(self, backbones: List[torch.nn.Module], heads: List[torch.nn.Module]):
        super().__init__()
        self.backbones = torch.nn.ModuleList(backbones)
        self.heads = torch.nn.ModuleList(heads)

    @property
    def n_basis(self) -> int:
        return len(self.backbones)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values: Optional[ValueBankCache] = None, use_cache=True):
        values, caches = [], []
        for i, (bb, head) in enumerate(zip(self.backbones, self.heads)):
            out = bb(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                     past_key_values=None if past_key_values is None else past_key_values.caches[i], use_cache=use_cache)
            values.append(head(out.last_hidden_state.to(head.weight.dtype)).squeeze(-1))      # (B, T): value AFTER each token
            caches.append(out.past_key_values)
        return torch.stack(values, dim=-1), ValueBankCache(caches)


def load_value_bank(value_dir: str, n_basis: int, device: str = "cuda", dtype=torch.bfloat16):
    """value_dir/basis0 .. basis{n_basis-1} (train_value_han.py --output-dir). Returns (ValueBank in eval mode, meta of basis 0)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    backbones, heads, meta0 = [], [], None
    for k in range(n_basis):
        d = os.path.join(value_dir, f"basis{k}")
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        meta0 = meta0 or meta
        assert meta["basis"] == k and meta["n_basis"] == n_basis, f"{d}: trained as basis {meta['basis']} of {meta['n_basis']}, expected {k} of {n_basis}"
        lm = AutoModelForCausalLM.from_pretrained(meta["backbone"], torch_dtype=dtype)
        lm = PeftModel.from_pretrained(lm, d).merge_and_unload()          # the LoRA was wrapped around the causal LM inside TRL's value-head model
        head_sd = torch.load(os.path.join(d, "pytorch_model.bin"), map_location="cpu")
        head = torch.nn.Linear(lm.config.hidden_size, 1)
        head.weight.data = head_sd["v_head.summary.weight"].float()
        head.bias.data = head_sd["v_head.summary.bias"].float()
        backbones.append(lm.model.to(device).eval())                       # the transformer without the LM head: only hidden states are needed
        heads.append(head.to(device).eval())
        del lm
    return ValueBank(backbones, heads).eval(), meta0
