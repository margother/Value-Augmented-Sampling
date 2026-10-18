"""Train ONE scalar value model per LoRe basis with Han et al.'s own VASTrainer (../vas_trainer.py, the unmodified upstream clone).

This is the published VAS recipe for several rewards (VAS paper, Sec. 4.2 "Composing multiple alignment objectives"): one value function per
reward, trained independently, combined only at inference as sum_k w_k * V_k (see decode_han.py). Here the K rewards are the LoRe basis rewards
b_k(x, y) = (e(x, y)^T V)_k recomputed from vas/collect_samples.py's stored embeddings (e^T V), so the SAME samples.pt serves every basis and every K. Reward given to the trainer =
b_k / --reward-scale (default 100 = LoRe's own logit scale: its preference logit is (e_c - e_r)^T V w_u / 100), i.e. rewards are in
Bradley-Terry-logit units and beta = 1 at decoding is the KL-regularised optimum at temperature 1 for the user's LoRe reward.

Follows tinyllama_hh.py (their entry point): pre-sampled (query, response) token ids + a scalar score per pair -> vas_trainer.step(queries, responses,
scores); value head zero-initialised, dropout off, LoRA on the backbone (bf16 instead of their 4-bit), their VASConfig defaults (lam=0.95, gamma=1,
vas_epochs=2, lr 1.41e-5). Only the data source and the reward differ. Runs in env_prlhf (trl 0.10.1). GPU work -- Slurm only.

    python vas_lore/train_value_han.py --samples .../vas_lore_k5/samples.pt --policy-path meta-llama/Llama-3.2-3B-Instruct \
        --backbone meta-llama/Llama-3.2-1B-Instruct --output-dir .../vas_han/value_models   # trains k = 0..K-1 one after the other
"""
import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the cloned repo root: vas_trainer.py / vas_config.py, UNMODIFIED
sys.path.insert(0, "/mnt/home/margot/personalization")


def encode(tokenizer, prompt, response):
    """(query ids = chat-templated prompt + generation prompt, response ids incl. the end-of-turn token)."""
    def ids(conv, **kw):
        out = tokenizer.apply_chat_template(conv, tokenize=True, **kw)
        return list(out["input_ids"] if isinstance(out, dict) else out)
    user = [{"role": "user", "content": prompt}]
    q = ids(user, add_generation_prompt=True)
    full = ids(user + [{"role": "assistant", "content": response}])
    n = 0
    while n < min(len(q), len(full)) and q[n] == full[n]:
        n += 1
    return q[:n], full[n:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True)
    p.add_argument("--lore-checkpoint-dir", default=None,
                   help="dir with lore_basis.safetensors: the basis rewards are recomputed as embeddings @ V for THIS checkpoint (any K) from the samples' stored embeddings")
    p.add_argument("--policy-path", default="meta-llama/Llama-3.2-3B-Instruct", help="used as VASTrainer's ref_model (as in their script)")
    p.add_argument("--backbone", default="meta-llama/Llama-3.2-1B-Instruct", help="value model; must share the policy's tokenizer")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bases", type=int, nargs="*", default=None, help="basis indices to train (default: all K)")
    p.add_argument("--reward-scale", type=float, default=100.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--mini-batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2, help="passes over the samples (their vas_epochs is separate: optimisation epochs inside step())")
    p.add_argument("--lr", type=float, default=1.41e-5)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--max-len", type=int, default=1536)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    from trl import AutoModelForCausalLMWithValueHead, set_seed
    from vas_config import VASConfig
    from vas_trainer import VASTrainer
    from dpo_LRF.timing_scope import TIMING_SCOPE, hardware_info

    data = torch.load(args.samples, map_location="cpu")
    if args.lore_checkpoint_dir:
        from safetensors.torch import load_file
        V = load_file(os.path.join(args.lore_checkpoint_dir, "lore_basis.safetensors"))["V"].float()      # (hidden, K)
        rewards = data["embeddings"].float() @ V                                                       # (N, K) raw LoRe basis rewards b = e^T V
    else:
        rewards = data["rewards"].float()                  # (N, K) raw LoRe basis rewards stored at collection time
    K = rewards.shape[1]
    tok = AutoTokenizer.from_pretrained(args.backbone)
    tok.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    items = []
    for i, (q, r) in enumerate(zip(data["prompts"], data["responses"])):
        qi, ri = encode(tok, q, r)
        if 0 < len(ri) and len(qi) + len(ri) <= args.max_len:
            items.append((torch.tensor(qi), torch.tensor(ri), i))
    print(f"=== {len(items)}/{len(data['prompts'])} samples kept (<= {args.max_len} tokens), K={K}, reward scale 1/{args.reward_scale:g} ===", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    for k in (args.bases if args.bases is not None else range(K)):
        out_k = os.path.join(args.output_dir, f"basis{k}")
        if os.path.exists(os.path.join(out_k, "meta.json")):
            print(f"=== basis {k}: already trained -- skipping ===", flush=True)
            continue
        set_seed(args.seed)
        cfg = VASConfig(model_name=args.backbone, ref_model_name=args.policy_path, batch_size=args.batch_size, mini_batch_size=args.mini_batch_size,
                        gradient_accumulation_steps=args.batch_size // args.mini_batch_size, learning_rate=args.lr, lam=args.lam, seed=args.seed)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(args.backbone, torch_dtype=torch.bfloat16)
        model.pretrained_model = get_peft_model(model.pretrained_model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, bias="none", task_type="CAUSAL_LM"))
        model.is_peft_model = True
        torch.nn.init.zeros_(model.v_head.summary.weight)
        torch.nn.init.zeros_(model.v_head.summary.bias)
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.policy_path, torch_dtype=torch.bfloat16)
        trainer = VASTrainer(cfg, model, ref_model, tok)

        t_train, n_steps, rng = 0.0, 0, random.Random(args.seed + k)
        for epoch in range(args.epochs):
            order = list(range(len(items)))
            rng.shuffle(order)
            for s in range(0, len(order) - args.batch_size + 1, args.batch_size):
                batch = [items[j] for j in order[s:s + args.batch_size]]
                scores = [(rewards[i, k] / args.reward_scale).reshape(()) for _, _, i in batch]
                t0 = time.time()
                stats = trainer.step([q for q, _, _ in batch], [r for _, r, _ in batch], scores)
                t_train += time.time() - t0
                n_steps += 1
                if n_steps % 20 == 0:
                    err = stats.get("vas/val/error", stats.get("val/error"))
                    print(f"  basis {k} epoch {epoch} step {n_steps} val/error {err} ({t_train:.0f}s)", flush=True)
        trainer.save_pretrained(out_k)
        with open(os.path.join(out_k, "meta.json"), "w") as f:
            json.dump({"basis": k, "n_basis": K, "reward_scale": args.reward_scale, "backbone": args.backbone, "policy": args.policy_path,
                       "lam": args.lam, "epochs": args.epochs, "n_samples": len(items), "n_optimizer_steps": n_steps, "training_time_seconds": t_train,
                       "reward_mean_raw": float(rewards[:, k].mean()), "reward_std_raw": float(rewards[:, k].std()),
                       "samples": os.path.abspath(args.samples), "lore_checkpoint_dir": args.lore_checkpoint_dir, "timing_scope": TIMING_SCOPE, "hardware": hardware_info()}, f, indent=2)
        print(f"=== basis {k} done: {n_steps} steps, {t_train:.0f}s ===", flush=True)
        del trainer, model, ref_model
        torch.cuda.empty_cache()
    print("[INFO] Script execution completed successfully.", flush=True)


if __name__ == "__main__":
    main()
