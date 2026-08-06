"""Generate BioTriplex-style gold JSONL for TREC-QC from the raw jsonl splits.

Output format (matches ``datasets/botriplex/Preprocessed BioTriplex/test_gold_general_qa.txt``):

    {doc_key, output='a)', input=<question text>, relation={...}, entities=[...],
     coarse_relation=<coarse text>, label_idx=<0..5>}

Usage::

    python trec_gold.py \
        --trec_dir /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/datasets/trec-qc \
        --output_dir /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/TrecAATestData/gold

Produces:

    test_gold_general_qa.txt  (500 samples — matches BioTriplex test count)
    val_gold_general_qa.txt   (543 samples — extra: validate split)
    train_gold_general_qa.txt (4909 samples — extra: train split for reference)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Coarse label 0..5 → answer letter a..f
COARSE_LETTERS = ["a", "b", "c", "d", "e", "f"]


def load_split(jsonl_path: Path) -> List[Dict]:
    out = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def to_gold_record(idx: int, item: Dict) -> Dict:
    """Convert a TREC-QC jsonl item to BioTriplex-style gold record."""
    coarse_idx = int(item["label_coarse"])
    coarse_text = item.get("label_coarse_text", "")
    coarse_code = item.get("label_coarse_original", "")
    fine_idx = int(item["label"])
    fine_text = item.get("label_text", "")
    fine_code = item.get("label_original", "")
    text = item["text"]

    return {
        "doc_key": f"trec_{idx}",
        "input": text,
        "output": f"{COARSE_LETTERS[coarse_idx]})",
        "relation": {
            "coarse": coarse_code,
            "coarse_text": coarse_text,
            "fine": fine_code,
            "fine_text": fine_text,
        },
        "entities": [],
        "coarse_relation": coarse_text,
        "label_idx": coarse_idx,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trec_dir", required=True,
                    help="Path to datasets/trec-qc/")
    ap.add_argument("--output_dir", required=True,
                    help="Output directory for *_gold_general_qa.txt")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="Which splits to export")
    args = ap.parse_args()

    trec_dir = Path(args.trec_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        in_path = trec_dir / f"{split}.jsonl"
        if not in_path.exists():
            print(f"[skip] {in_path} not found", file=sys.stderr)
            continue
        records = load_split(in_path)
        out_path = output_dir / f"{split}_gold_general_qa.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for i, item in enumerate(records):
                f.write(json.dumps(to_gold_record(i, item), ensure_ascii=False) + "\n")
        # Quick stats
        from collections import Counter
        ctr = Counter(int(r["label_coarse"]) for r in records)
        print(f"[ok] {split}: {len(records)} samples → {out_path}")
        for k in sorted(ctr.keys()):
            print(f"     class {k}: {ctr[k]}")


if __name__ == "__main__":
    main()