#!/usr/bin/env python3
"""
ner_infer.py — BioTriplex NER 推理，生成 JSON span 输出。

本脚本绕过原版 recipes/quickstart/inference/local_inference/inference.py：
  - 原版的 `assert not rel_dataset` 是死引用（papers repo 自己有 bug）
  - 原版用 gradio/CLI 模式，没法 saved outputs

策略：跟 classification_genrel/infer_and_save.py 同套风格：
  - 读 ${DATA_PATH}/test_para.txt
  - 用 BioTriplexNERDataset 构 test split
  - 把 PEFT adapter 装到 base model
  - 对每个 doc_key 跑一次 generate，得到原始输出文本
  - 输出 JSON: {doc_key: {"output": "...", "raw_text": "..."}}
"""
import argparse
import json
import sys
from pathlib import Path

# Bootstrap compat shims (inference-side mapping fixups).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "llama-rec" / "_compat"))
try:
    import infer_compat  # noqa: F401
    infer_compat.install()
except Exception as _e:
    print(f"[infer-compat] WARNING: {type(_e).__name__}: {_e}", file=sys.stderr)

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--peft_model", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--max_eval_samples", type=int, default=-1)
    p.add_argument("--max_new_tokens", type=int, default=2000)
    p.add_argument("--temperature", type=float, default=0.6)
    return p.parse_args()


def build_config(args):
    class _C:
        pass
    cfg = _C()
    cfg.data_path = args.data_path + "/"
    cfg.use_entity_tokens_as_targets = False
    cfg.entity_special_tokens = False
    cfg.bidirectional_attention_in_entity_tokens = False
    cfg.shift_entity_tokens = False
    cfg.upweight_minority_class = False
    return cfg


def load_dataset(args, tokenizer):
    """用 BioTriplexNERDataset 实例化 test split。"""
    from llama_recipes.datasets.biotriplex_ner_dataset import BioTriplexNERDataset
    cfg = build_config(args)
    dataset = BioTriplexNERDataset(cfg, tokenizer, "test", max_words=None)
    return dataset


def collect_prompts_and_gold(dataset):
    """Use the dataset's own prompt builder so we get exactly the same prompt
    format as training (and as the broken inference.py expected)."""
    if hasattr(dataset, "get_all_input_prompts"):
        prompts = dataset.get_all_input_prompts(bidirectional=False)
    else:
        prompts = {}
        for idx in range(len(dataset)):
            item = dataset[idx]
            key = item.get("doc_key") or str(idx)
            prompts[key] = item.get("input") or ""
    gold_outputs = {}
    for item in dataset.data:
        gold_outputs[item.get("doc_key")] = item.get("output", "")
    return prompts, gold_outputs


def main():
    args = parse_args()

    print(f"[INFO] loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("[INFO] building test dataset (writes gold file as side effect)")
    dataset = load_dataset(args, tokenizer)

    prompts, gold = collect_prompts_and_gold(dataset)
    print(f"[INFO] num prompts: {len(prompts)}")

    if args.max_eval_samples > 0:
        keys = sorted(prompts.keys())[: args.max_eval_samples]
        prompts = {k: prompts[k] for k in keys}
        gold = {k: gold[k] for k in keys}
        print(f"[INFO] truncated to {len(prompts)} samples for dev")

    print(f"[INFO] loading base model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
    )
    if args.peft_model:
        print(f"[INFO] loading PEFT adapter: {args.peft_model}")
        model = PeftModel.from_pretrained(model, args.peft_model)
    model.eval()

    outputs = {}
    for doc_key, prompt_str in tqdm(prompts.items(), desc="ner-infer"):
        if not prompt_str:
            outputs[doc_key] = ""
            continue
        inputs = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                top_p=1.0,
                top_k=200,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = gen[0, inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        outputs[doc_key] = raw.strip()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"[INFO] wrote {len(outputs)} entries to {out_path}")


if __name__ == "__main__":
    main()
