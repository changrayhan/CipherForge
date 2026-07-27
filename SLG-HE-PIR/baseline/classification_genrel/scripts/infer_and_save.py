#!/usr/bin/env python3
"""
infer_and_save.py — BioTriplex GenRel QA 推理，保存 7 类选项的 logits。

用途：
  - 论文原 inference.py 只生成 'a)'/'b)'/.../'g)' 文本，无法获得 7 类概率
  - 本脚本读取 BioTriplexQAKShotDataset 构造的 prompt，
    对每个 prompt 做单次前向，取最后一个 token 位置在
    a)/b)/.../g) 7 个 token id 上的 logits，softmax 后作为 7 类概率
  - 同时输出贪婪生成的字母（用于 Macro F1 / 多标签 F1）

输出 JSON 格式：
  {doc_key: {"answer": "a)", "logits": [..7..], "probs": [..7..], "predicted_relation": "..."}}
"""

import argparse
import json
import sys
from pathlib import Path

# Bootstrap compat shims (inference-side mapping fixups).
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "llama-rec" / "_compat"))
try:
    import infer_compat  # noqa: F401
    infer_compat.install()
except Exception as _e:
    print(f"[infer-compat] WARNING: {type(_e).__name__}: {_e}", file=sys.stderr)

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


OPTION_LETTERS = ["a", "b", "c", "d", "e", "f", "g"]
GENERAL_RELATIONS = [
    "pathological",
    "modulatory",
    "expression change",
    "diagnosis",
    "therapy",
    "no relation",
    "relation undefined",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--peft_model", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--quantization", default=None)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--max_eval_samples", type=int, default=-1,
                   help="开发期限制样本数；-1 表示全量")
    return p.parse_args()


def get_option_token_ids(tokenizer, letters=OPTION_LETTERS):
    """每个字母映射到一个 token id。

    策略：把 'a)' / 'a' / ' a)' / ' a' 四个候选都试一下，取第一个能编码成单个 token 的。
    """
    token_ids = []
    for letter in letters:
        candidates = [f"{letter})", letter, f" {letter})", f" {letter}"]
        chosen = None
        for c in candidates:
            ids = tokenizer.encode(c, add_special_tokens=False)
            if len(ids) == 1:
                chosen = ids[0]
                break
        if chosen is None:
            ids = tokenizer.encode(candidates[0], add_special_tokens=False)
            chosen = ids[0]
        token_ids.append(chosen)
    return token_ids


def build_config(args):
    class _C:
        pass
    cfg = _C()
    cfg.data_path = args.data_path + "/"
    cfg.use_entity_tokens_as_targets = False
    cfg.entity_special_tokens = False
    cfg.upweight_minority_class = False
    cfg.bidirectional_attention_in_entity_tokens = False
    cfg.shift_entity_tokens = False
    cfg.num_of_shots = 0
    cfg.general_relations = True
    cfg.group_relations = False
    cfg.max_words = None
    cfg.return_neg_relations = False
    return cfg


def build_prompts(args, tokenizer):
    from llama_recipes.datasets.biotriplex_qakshot_dataset import BioTriplexQADataset

    cfg = build_config(args)
    dataset = BioTriplexQADataset(cfg, tokenizer, "test", max_words=None)
    if hasattr(dataset, "get_all_input_prompts"):
        prompts = dataset.get_all_input_prompts(bidirectional=False)
    else:
        prompts = {}
        for idx in range(len(dataset)):
            item = dataset[idx]
            prompts[item.get("doc_key", str(idx))] = item.get("input", "")
    return prompts


def main():
    args = parse_args()

    print(f"[INFO] loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    option_token_ids = get_option_token_ids(tokenizer)
    print(f"[INFO] option token mapping: {list(zip(OPTION_LETTERS, option_token_ids))}")

    prompts = build_prompts(args, tokenizer)
    print(f"[INFO] num prompts: {len(prompts)}")

    if args.max_eval_samples > 0:
        keys = sorted(prompts.keys())[: args.max_eval_samples]
        prompts = {k: prompts[k] for k in keys}
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
    for doc_key, prompt in tqdm(prompts.items(), desc="infer"):
        prompt_str = prompt["prompt"] if isinstance(prompt, dict) else prompt
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        last_logits = out.logits[0, -1, :].float().cpu()
        option_logits = [float(last_logits[tid]) for tid in option_token_ids]
        probs = torch.softmax(torch.tensor(option_logits), dim=0).tolist()
        best_idx = int(torch.tensor(probs).argmax().item())
        answer_letter = OPTION_LETTERS[best_idx]
        outputs[doc_key] = {
            "answer": f"{answer_letter})",
            "logits": option_logits,
            "probs": probs,
            "predicted_relation": GENERAL_RELATIONS[best_idx],
        }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"[INFO] wrote {len(outputs)} entries to {out_path}")


if __name__ == "__main__":
    main()