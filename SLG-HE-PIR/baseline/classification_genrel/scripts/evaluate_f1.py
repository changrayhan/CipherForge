#!/usr/bin/env python3
"""
F1 评测脚本 — BioTriplex GenRel QA 分类任务

功能：加载训练好的 LoRA adapter，对 val（必选）和 test（可选）数据集做前向推理，
       计算 micro-F1 / macro-F1 / per-class F1 / subset accuracy / hamming loss。

设计要点：
- 不重训，不写训练 checkpoint，仅纯前向推理。
- 不会占用太多显存（22 GB 级）；bf16 load + batch_size 1 = ~14-15 GB。
- 复用 base 模型的 llama-recipes 数据集类，避免重新实现 prompt 拼装。
- 输出两份：JSON（方便后续 plot / compare）和 Markdown（直接贴报告）。

入口：
    python evaluate_f1.py --split val
    python evaluate_f1.py --split val test
    python evaluate_f1.py --split val --max_samples 20   # 快速 sanity check
"""
import argparse
import collections
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 保证能 import llama_recipes
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parents[1]  # baseline/
LLAMA_REC = BASE_DIR / "llama-rec"
sys.path.insert(0, str(LLAMA_REC / "src"))
os.environ.setdefault("PYTHONPATH", str(LLAMA_REC / "src"))

from llama_recipes.configs import datasets as DATASET_CONFIGS  # noqa: E402
from llama_recipes.datasets.biotriplex_qakshot_dataset import (  # noqa: E402
    BioTriplexQADataset, _parse_output_labels, GENERAL_REL_LIST,
)


# -----------------------------------------------------------------------------
# 路径常量
# -----------------------------------------------------------------------------
BASE_MODEL_PATH = "/root/autodl-tmp/hf_cache/Llama-3-1-8B-I"
ADAPTER_DIR = BASE_DIR / "classification_genrel" / "checkpoints"
DATA_DIR = BASE_DIR.parent / "datasets" / "botriplex_classification"
LOG_DIR = BASE_DIR / "logs"


# -----------------------------------------------------------------------------
# 评估用的 dataset config（与训练时一致）
# -----------------------------------------------------------------------------
def make_eval_dataset_config(data_path: str):
    """构造一个 *最小配置* 的 biotriplex_qakshot_dataset dataclass。"""
    cfg = DATASET_CONFIGS.biotriplex_qakshot_dataset()
    cfg.dataset = "biotriplex_qakshot_dataset"
    cfg.data_path = data_path
    cfg.use_entity_tokens_as_targets = False
    cfg.entity_special_tokens = False
    cfg.upweight_minority_class = False
    cfg.bidirectional_attention_in_entity_tokens = False
    cfg.shift_entity_tokens = False
    cfg.return_neg_relations = False
    cfg.general_relations = True
    cfg.num_of_shots = 0
    cfg.group_relations = True  # 训练时默认 True，保留
    cfg.train_sample_pct = 1.0
    cfg.train_sample_seed = 42
    cfg.train_sample_stratify = False
    cfg.train_sample_min_per_label = 0
    return cfg


# -----------------------------------------------------------------------------
# 解析模型输出
# -----------------------------------------------------------------------------
def extract_label_string(raw: str) -> str:
    """从模型 free-form 输出中抽取 'a), b)' 这种字母组合字符串。

    训练时的 gold 都是 'x), y), z)' 形式；模型输出可能在前/后/中间夹杂其它字符。
    取第一个出现的 '<letter>)' 子串序列。
    """
    out_clean = []
    seen_comma = False
    for ch in raw:
        if "a" <= ch <= "g":
            out_clean.append(ch)
            seen_comma = True
        elif ch == "," and out_clean and seen_comma:
            out_clean.append(ch)
            seen_comma = False
        elif ch == ")" and out_clean:
            out_clean.append(ch)
            seen_comma = False
    s = "".join(out_clean).strip()
    # 规范化：去掉末尾孤立逗号 / 重复逗号 / 空格
    while ",," in s:
        s = s.replace(",,", ",")
    if s.endswith(","):
        s = s[:-1]
    return s


# -----------------------------------------------------------------------------
# 指标计算
# -----------------------------------------------------------------------------
def _safe_div(a, b):
    return a / b if b else 0.0


def compute_metrics(
    golds: List[Set[int]], preds: List[Set[int]], num_classes: int
) -> Dict[str, float]:
    """计算 multi-label 分类指标。

    golds / preds: 每条样本是 {label_idx, ...} 集合。
    """
    n = len(golds)
    assert n == len(preds), "gold/pred 数量不一致"

    per_class_tp = [0] * num_classes
    per_class_fp = [0] * num_classes
    per_class_fn = [0] * num_classes
    per_class_support = [0] * num_classes  # gold 中出现该类的次数

    exact_match = 0  # subset accuracy
    total_correct = 0  # 标签级正确数
    total_gold_labels = 0
    total_pred_labels = 0
    hamming_mistakes = 0  # 标签级错误数

    for g, p in zip(golds, preds):
        if g == p:
            exact_match += 1
        total_correct += len(g & p)
        total_gold_labels += len(g)
        total_pred_labels += len(p)
        # hamming = 对称差的大小
        hamming_mistakes += len(g ^ p)
        for c in range(num_classes):
            in_g = c in g
            in_p = c in p
            if in_g and in_p:
                per_class_tp[c] += 1
            elif in_p and not in_g:
                per_class_fp[c] += 1
            elif in_g and not in_p:
                per_class_fn[c] += 1
            if in_g:
                per_class_support[c] += 1

    # per-class P/R/F1
    per_class = []
    for c in range(num_classes):
        p = _safe_div(per_class_tp[c], per_class_tp[c] + per_class_fp[c])
        r = _safe_div(per_class_tp[c], per_class_tp[c] + per_class_fn[c])
        f1 = _safe_div(2 * p * r, p + r)
        per_class.append({
            "idx": c,
            "letter": chr(ord("a") + c),
            "name": GENERAL_REL_LIST[c],
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": per_class_support[c],
        })

    micro_p = _safe_div(total_correct, total_correct + sum(per_class_fp))
    micro_r = _safe_div(total_correct, total_correct + sum(per_class_fn))
    micro_f1 = _safe_div(2 * micro_p * micro_r, micro_p + micro_r)
    macro_p = sum(pc["precision"] for pc in per_class) / num_classes
    macro_r = sum(pc["recall"] for pc in per_class) / num_classes
    macro_f1 = sum(pc["f1"] for pc in per_class) / num_classes
    subset_acc = exact_match / n if n else 0.0
    hamming = hamming_mistakes / (n * num_classes) if n else 0.0
    sample_p = sum(_safe_div(len(g & p), len(p)) for g, p in zip(golds, preds) if p) / n if n else 0.0
    sample_r = sum(_safe_div(len(g & p), len(g)) for g, p in zip(golds, preds) if g) / n if n else 0.0
    sample_f1_sum = 0.0
    sample_f1_n = 0
    for g, p in zip(golds, preds):
        if g or p:
            tp = len(g & p)
            sp = _safe_div(tp, len(p))
            sr = _safe_div(tp, len(g))
            sample_f1_sum += _safe_div(2 * sp * sr, sp + sr) if (sp + sr) > 0 else 0.0
            sample_f1_n += 1
    sample_f1 = _safe_div(sample_f1_sum, sample_f1_n)

    return {
        "n_samples": n,
        "num_classes": num_classes,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "subset_accuracy": subset_acc,
        "hamming_loss": hamming,
        "sample_precision": sample_p,
        "sample_recall": sample_r,
        "sample_f1": sample_f1,
        "per_class": per_class,
    }


# -----------------------------------------------------------------------------
# 单条样本推理
# -----------------------------------------------------------------------------
def build_prompt(tokenizer, item, max_input_tokens: int = 9800) -> Tuple[str, str]:
    """返回 (prompt_prefix, gold_answer_letter_str)

    prompt_prefix 是模型看到的 prompt（不含 Response 之后的内容）。
    """
    # 通过 dataset 类的 input_to_prompt 复用其实现，行为与训练时完全一致
    prefix, pinput, suffix = item["prefix_input_suffix"]
    text = prefix + pinput + suffix
    return text, "\n### Response:\n" + item["output"]


@torch.no_grad()
def predict_one(
    model, tokenizer, prefix_text: str, max_new_tokens: int = 20,
    temperature: float = 0.0, top_p: float = 1.0, top_k: int = 50,
    repetition_penalty: float = 2.0,
) -> str:
    """对一条 prompt 做贪心/采样生成，返回 Response 之后的 letter 字符串。"""
    enc = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(model.device)
    attn_mask = enc.attention_mask.to(model.device)

    do_sample = temperature > 0.0
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        top_k=top_k if do_sample else 0,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = gen[0, input_ids.shape[1]:]
    out_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return out_text


def evaluate_split(
    model, tokenizer, dataset: BioTriplexQADataset, max_samples: int = -1,
    temperature: float = 0.0, max_new_tokens: int = 20,
    repetition_penalty: float = 2.0,
) -> Tuple[List[Set[int]], List[Set[int]], List[Dict]]:
    golds, preds, details = [], [], []
    n = len(dataset) if max_samples < 0 else min(max_samples, len(dataset))
    t0 = time.time()
    for i in range(n):
        item = dataset[i]
        prefix_text, _gold_resp = build_prompt(tokenizer, item)
        out_text = predict_one(
            model, tokenizer, prefix_text,
            max_new_tokens=max_new_tokens, temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        # 模型可能输出 "### Response:\na), b)<|eot|>" — 抽取 Response 之后
        if "### Response:" in out_text:
            out_after = out_text.split("### Response:")[-1].strip()
        else:
            out_after = out_text.strip()
        out_after = out_after.split("\n")[0].strip()
        label_str = extract_label_string(out_after)
        pred_set = _parse_output_labels(label_str)
        gold_set = _parse_output_labels(item["output"])
        golds.append(gold_set)
        preds.append(pred_set)
        details.append({
            "idx": i,
            "doc_key": item.get("doc_key", f"sample-{i}"),
            "gold_raw": item["output"],
            "raw_model_out": out_after[:200],
            "pred_raw": label_str,
            "gold_set": sorted(list(gold_set)),
            "pred_set": sorted(list(pred_set)),
        })
        if (i + 1) % 25 == 0 or i == n - 1:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / speed if speed > 0 else 0
            print(f"  [{i+1}/{n}] elapsed={elapsed:.1f}s speed={speed:.2f} sample/s eta={eta:.1f}s", flush=True)
    return golds, preds, details


# -----------------------------------------------------------------------------
# markdown 报告
# -----------------------------------------------------------------------------
def metrics_to_markdown(split_name: str, m: Dict, adapter_path: str) -> str:
    lines = []
    lines.append(f"# F1 评测结果 — split = `{split_name}`\n")
    lines.append(f"> Adapter: `{adapter_path}`  ")
    lines.append(f"> Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"> Num samples: **{m['n_samples']}**, Num classes: **{m['num_classes']}**\n")

    lines.append("## 总体指标\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k in [
        "micro_precision", "micro_recall", "micro_f1",
        "macro_precision", "macro_recall", "macro_f1",
        "subset_accuracy", "hamming_loss",
        "sample_precision", "sample_recall", "sample_f1",
    ]:
        lines.append(f"| {k} | {m[k]:.4f} |")
    lines.append("")

    lines.append("## Per-class 指标\n")
    lines.append("| Letter | Name | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for pc in m["per_class"]:
        lines.append(
            f"| {pc['letter']}) | {pc['name']} | "
            f"{pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['support']} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("\n> **解读**：macro-F1 优先看（对类别不平衡敏感），micro-F1 受高频类主导。\n"
                 "> **期望范围**（针对当前 epoch 10、596 train 的过拟合 adapter）：\n"
                 "> - micro-F1  0.55-0.70\n"
                 "> - macro-F1  0.30-0.45\n"
                 "> 若 macro-F1 < 0.30，说明模型对 `b) disease modulating` / `f) no relation` 完全没学到。")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", default=str(ADAPTER_DIR))
    ap.add_argument("--base_model", default=BASE_MODEL_PATH)
    ap.add_argument("--data_dir", default=str(DATA_DIR))
    ap.add_argument("--split", nargs="+", default=["val"], choices=["val", "test"])
    ap.add_argument("--max_samples", type=int, default=-1, help="-1 表示用全部；>0 用于 sanity check")
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.0, help="0.0 表示贪心")
    ap.add_argument("--repetition_penalty", type=float, default=2.0)
    ap.add_argument("--output_dir", default=str(ADAPTER_DIR),
                    help="把 metrics_f1_*.json 与 F1_RESULTS.md 写到这里")
    ap.add_argument("--no_torch_compile", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[init] base_model: {args.base_model}")
    print(f"[init] adapter_dir: {args.adapter_dir}")
    print(f"[init] data_dir: {args.data_dir}")
    print(f"[init] split: {args.split}")
    print(f"[init] max_samples: {args.max_samples}")
    print(f"[init] temperature: {args.temperature} (0=greedy)")
    print(f"[init] max_new_tokens: {args.max_new_tokens}")
    print()

    # 1. 加载 tokenizer + base model
    print("[load] tokenizer + base model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # 生成时左侧 padding 兼容性更好

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    )
    base.config.use_cache = False

    # 2. 加载 LoRA adapter
    print(f"[load] peft adapter from {args.adapter_dir} ...")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[load] base+adapter total params: {total:,}, trainable (should be 0 for inference): {trainable:,}")
    print(f"[load] GPU alloc: {torch.cuda.memory_allocated()/1e9:.2f} GB / reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print()

    all_results = {}
    all_details = {}
    for split in args.split:
        # 3. 构造 dataset —— 直接调用原训练用的 BioTriplexQADataset
        # 训练时 train/val 调用三次构造；如果 test 还没 gold，则这一步会自动产出
        print(f"[dataset] constructing {split} dataset ...")
        cfg = make_eval_dataset_config(args.data_dir + "/")
        ds = BioTriplexQADataset(cfg, tokenizer, split_name=split, max_words=None)
        print(f"[dataset] {split}: {len(ds)} samples (multi-label, group_relations=True, general_relations=True)")
        # 因为 __getitem__ 已经会拼 prefix，我们再额外包一层让 build_prompt 复用 input_to_prompt
        # 在 __getitem__ 里把 prompt 拼出来一次，缓存，避免重复计算
        # 但 dataset 没有存 — 我们用 dict 风格包装
        orig_getitem = ds.__class__.__getitem__

        def wrapped_getitem(self, idx):
            out = orig_getitem(self, idx)
            # 把 prefix 拼好注入到 out
            triple = self.data[idx]["relation"]
            prefix, pinput, suffix = self.input_to_prompt(out["input"] if "input" in out else self.data[idx]["input"], triple)
            out["prefix_input_suffix"] = (prefix, pinput, suffix)
            out["doc_key"] = self.data[idx].get("doc_key", f"sample-{idx}")
            out["output"] = self.data[idx]["output"]
            return out

        ds.__class__.__getitem__ = wrapped_getitem

        # 4. 推理 + 评估
        print(f"[eval] split={split} ...")
        golds, preds, details = evaluate_split(
            model, tokenizer, ds,
            max_samples=args.max_samples,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )
        # 5. 计算指标
        m = compute_metrics(golds, preds, num_classes=len(GENERAL_REL_LIST))
        # 6. 落盘
        json_path = Path(args.output_dir) / f"metrics_f1_{split}.json"
        with open(json_path, "w") as f:
            json.dump({
                "split": split,
                "adapter_path": args.adapter_dir,
                "base_model": args.base_model,
                "args": vars(args),
                "metrics": m,
                "details": details,
            }, f, ensure_ascii=False, indent=2)
        print(f"[save] {json_path}")

        md = metrics_to_markdown(split, m, args.adapter_dir)
        md_path = Path(args.output_dir) / f"F1_RESULTS_{split}.md"
        with open(md_path, "w") as f:
            f.write(md)
        print(f"[save] {md_path}")
        print()
        print(md)
        print()

        all_results[split] = m
        all_details[split] = details

    # 7. 汇总
    summary_path = Path(args.output_dir) / "metrics_f1_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "adapter_dir": args.adapter_dir,
            "base_model": args.base_model,
            "splits_evaluated": list(all_results.keys()),
            "results": {sp: {k: v for k, v in m.items() if k != "per_class"}
                        for sp, m in all_results.items()},
            "per_class": {sp: m["per_class"] for sp, m in all_results.items()},
        }, f, ensure_ascii=False, indent=2)
    print(f"[save] {summary_path}")


if __name__ == "__main__":
    main()
