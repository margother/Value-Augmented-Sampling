LoRe + VAS adapter for the PersonalLLM/PRISM/CA experiments (dpo_LRF project). Everything OUTSIDE this folder is the unmodified upstream clone of
https://github.com/idanshen/Value-Augmented-Sampling (same layout as Personalized_RLHF/prlhf_personalllm and fewshot-preference-optimization/fspo_personalllm).
Runs in env_prlhf (upstream trl 0.10.1 satisfies all of vas_trainer.py's imports; the README's private TRL fork is not publicly reachable).
  train_value_han.py  one scalar value model per LoRe basis, driven through upstream VASTrainer.step()
  decode_han.py       loads the K value models as ONE callable (ValueBank) for dpo_LRF/vas/vas_decoding.py's vas_generate (Han's decoding rule on transformers-4.5x Cache objects: their
                      VASLogitsProcessor needs the authors' TRL fork + tuple caches); the user weighting sum_k w_u,k * V_k happens there. Memory: ~2.5 GB per basis -> K <= 8 on one A6000.
Launchers and the generation manifest live in dpo_LRF (scripts/prism/LoRe+VAS/, generation_manifest/).
