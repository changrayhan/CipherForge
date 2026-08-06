"""Resplit BioTriplex classification data: merge train+val, then re-split 80/20.

Source:
    datasets/botriplex_classification/{train,val}_para.txt
    datasets/botriplex_classification/{train,val}_gold_general_grouped_qa.txt

Destination (under test-data/BioTriplex1BTestData/data/):
    train_para.txt      (80% of merged docs)
    val_para.txt        (20% of merged docs)
    train_gold_general_grouped_qa.txt   (re-derived from train_para via BioTriplexQADatasetClassification)
    val_gold_general_grouped_qa.txt     (re-derived from val_para)
    test_para.txt       (verbatim copy of original test_para.txt — we don't touch the test set)
    test_gold_general_grouped_qa.txt   (re-derived from test_para)

The 20% val split is **stratified by gold relation label**: we sample 20% of the
gold samples for each coarse relation class (a..g) so that rare classes (e.g.
"no relation", "relation undefined") are still represented in val.

The split is deterministic — uses a fixed random seed (42) so re-runs are
reproducible. We also write a `split_manifest.json` recording which doc_keys
went into train vs val.
"""

from __future__ import annotations

import collections
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path("/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR")
sys.path.insert(0, str(REPO_ROOT))

from src.data.biotriplex_dataset import build_biotriplex_dataset  # noqa: E402

SOURCE_DIR = REPO_ROOT / "datasets" / "botriplex_classification"
DEST_DIR = REPO_ROOT / "test-data" / "BioTriplex1BTestData" / "data"
TEST_FILE = "test_para.txt"
TRAIN_FILE = "train_para.txt"
VAL_FILE = "val_para.txt"


def load_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stratified_split_by_label(
    train_gold: List[dict],
    val_gold: List[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[List[str], List[str]]:
    """Return (train_doc_keys, val_doc_keys) by stratified sampling.

    Strategy:
        1. Concatenate train+val gold samples.
        2. Group gold samples by coarse label (a..g).
        3. For each label class, sample 20% of its doc_keys for validation.
        4. Combine: train = all_doc_keys - val_doc_keys.

    This guarantees that every coarse class has val samples (the original val
    was missing many classes — e.g. class b "modulatory" had 0 val samples).
    """
    rng = random.Random(seed)
    all_gold = train_gold + val_gold

    # Group doc_keys by coarse label
    by_label: Dict[str, set] = collections.defaultdict(set)
    for sample in all_gold:
        label = sample.get("output", "?")  # "a)", "b)", etc.
        if not label or label == "?":
            continue
        # Use the parent doc_key (strip _sentence_X_gene_Y_disease_Z_rel_W suffix)
        full_dk = sample["doc_key"]
        parent_dk = full_dk.split("_sentence_")[0]
        by_label[label].add(parent_dk)

    print(f"[stratify] merged gold samples: {len(all_gold)}")
    print(f"[stratify] coarse class doc_key counts:")
    for lbl in sorted(by_label.keys()):
        print(f"           {lbl}: {len(by_label[lbl])} unique doc_keys")

    val_doc_keys: set = set()
    for label, docs in by_label.items():
        docs_list = sorted(docs)
        n_val = max(1, int(round(len(docs_list) * val_ratio)))
        # Random sample without replacement
        sampled = rng.sample(docs_list, n_val)
        val_doc_keys.update(sampled)
        print(f"           {label}: {n_val}/{len(docs_list)} → val")

    all_doc_keys = set()
    for docs in by_label.values():
        all_doc_keys.update(docs)

    train_doc_keys = all_doc_keys - val_doc_keys
    print(f"\n[stratify] result: {len(train_doc_keys)} train, {len(val_doc_keys)} val")
    return sorted(train_doc_keys), sorted(val_doc_keys)


def main():
    print(f"SOURCE_DIR: {SOURCE_DIR}")
    print(f"DEST_DIR:   {DEST_DIR}\n")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: load train+val gold for stratification
    print("=== Step 1: load source data ===")
    train_para = load_jsonl(SOURCE_DIR / TRAIN_FILE)
    val_para = load_jsonl(SOURCE_DIR / VAL_FILE)
    train_gold = load_jsonl(SOURCE_DIR / "train_gold_general_grouped_qa.txt")
    val_gold = load_jsonl(SOURCE_DIR / "val_gold_general_grouped_qa.txt")
    print(f"  train_para: {len(train_para)} docs")
    print(f"  val_para:   {len(val_para)} docs")
    print(f"  train_gold: {len(train_gold)} samples")
    print(f"  val_gold:   {len(val_gold)} samples\n")

    # Step 2: stratified split
    print("=== Step 2: stratified 80/20 split ===")
    train_dks, val_dks = stratified_split_by_label(train_gold, val_gold)
    train_dk_set, val_dk_set = set(train_dks), set(val_dks)

    # Step 3: filter _para.txt records by doc_key
    print("\n=== Step 3: split _para.txt by doc_key ===")
    train_para_new = [d for d in (train_para + val_para) if d["doc_key"] in train_dk_set]
    val_para_new = [d for d in (train_para + val_para) if d["doc_key"] in val_dk_set]
    print(f"  new train_para: {len(train_para_new)} docs")
    print(f"  new val_para:   {len(val_para_new)} docs")

    # Step 4: copy test_para unchanged
    print("\n=== Step 4: copy test_para unchanged ===")
    shutil.copy2(SOURCE_DIR / TEST_FILE, DEST_DIR / TEST_FILE)
    print(f"  copied {TEST_FILE}")

    # Step 5: write split _para.txt
    print("\n=== Step 5: write split _para.txt ===")
    write_jsonl(DEST_DIR / TRAIN_FILE, train_para_new)
    write_jsonl(DEST_DIR / VAL_FILE, val_para_new)
    print(f"  wrote {DEST_DIR / TRAIN_FILE}")
    print(f"  wrote {DEST_DIR / VAL_FILE}")

    # Step 6: regenerate gold via BioTriplexQADatasetClassification (write_gold=True)
    print("\n=== Step 6: regenerate gold via build_biotriplex_dataset ===")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/CipherForgeCode/hf_cache/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for split in ("train", "val", "test"):
        build_biotriplex_dataset(
            task="classification",
            data_dir=str(DEST_DIR),
            tokenizer=tokenizer,
            split=split,
            max_length=4096,
            return_neg_relations=True,  # keep neg samples for richer training signal
        )

    # Step 7: write split_manifest.json
    print("\n=== Step 7: write split_manifest.json ===")
    manifest = {
        "source_train_docs": len(train_para),
        "source_val_docs": len(val_para),
        "new_train_docs": len(train_para_new),
        "new_val_docs": len(val_para_new),
        "val_ratio": 0.20,
        "seed": 42,
        "train_doc_keys": train_dks,
        "val_doc_keys": val_dks,
    }
    with open(DEST_DIR / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  wrote split_manifest.json")

    print("\n=== Done ===")
    print(f"All files under: {DEST_DIR}")


if __name__ == "__main__":
    main()
