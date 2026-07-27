#!/usr/bin/env python3
"""分析 BioTriplex 数据集的序列长度分布"""

import json
import os
import sys

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# 路径设置
DATASETS_DIR = "/root/autodl-tmp/SLG-HE-PIR/datasets"

def load_tokenizer():
    """加载 Llama tokenizer"""
    if HAS_TRANSFORMERS:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "/root/autodl-tmp/hf_cache/Llama-3-1-8B-I/",
                trust_remote_code=True,
                use_fast=True
            )
            print("Tokenizer: HuggingFace Llama-3.1-8B-Instruct")
            return tokenizer
        except Exception as e:
            print(f"Warning: Cannot load HF tokenizer: {e}")
    
    # 备用: 使用字符数估算 (1 token ≈ 4 chars)
    print("Tokenizer: Fallback (char/4 estimation)")
    return None

def count_tokens(text, tokenizer):
    """计算文本的 token 数量"""
    if tokenizer is None:
        return len(text) // 4
    return len(tokenizer.encode(text, add_special_tokens=False))

def get_text_from_sample(data):
    """从样本中提取文本内容"""
    # text 字段是列表
    if 'text' in data and isinstance(data['text'], list):
        return ' '.join(data['text'])
    return ''

def analyze_file(filepath, tokenizer):
    """分析单个文件"""
    lengths = []
    max_sample = None
    max_len = 0
    total_samples = 0
    
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                text = get_text_from_sample(data)
                
                if text:
                    total_samples += 1
                    token_len = count_tokens(text, tokenizer)
                    lengths.append(token_len)
                    
                    if token_len > max_len:
                        max_len = token_len
                        max_sample = {
                            'file': os.path.basename(filepath),
                            'line': line_num,
                            'doc_key': data.get('doc_key', 'N/A'),
                            'token_length': token_len,
                            'char_length': len(text),
                            'text_preview': text[:500] + '...' if len(text) > 500 else text
                        }
            except json.JSONDecodeError:
                continue
    
    if not lengths:
        return None
    
    return {
        'file': os.path.basename(filepath),
        'path': filepath,
        'total_samples': total_samples,
        'valid_samples': len(lengths),
        'min': min(lengths),
        'max': max(lengths),
        'avg': sum(lengths) / len(lengths),
        'max_sample': max_sample,
        'all_lengths': lengths
    }

def print_result(res):
    """打印单个文件的结果"""
    if res is None:
        return
    
    print(f"\n{'─'*80}")
    print(f"📄 {res['file']}")
    print(f"{'─'*80}")
    print(f"  样本数: {res['valid_samples']} (总行数: {res['total_samples']})")
    print(f"  Token 长度:")
    print(f"    最小: {res['min']}")
    print(f"    最大: {res['max']} ← 最长!")
    print(f"    平均: {res['avg']:.1f}")
    
    if res['max_sample']:
        s = res['max_sample']
        print(f"\n  最长样本详情:")
        print(f"    doc_key: {s['doc_key']}")
        print(f"    token_length: {s['token_length']}, char_length: {s['char_length']}")
        print(f"    文本预览:\n    {s['text_preview']}")

def main():
    print("=" * 80)
    print("🔍 BioTriplex 数据集序列长度分析")
    print("=" * 80)
    
    tokenizer = load_tokenizer()
    
    # =========================================================================
    # 任务 1: GenRel QA (分类任务) - 使用 *_para.txt
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("📊 任务 A: GenRel QA (分类任务) - *_para.txt")
    print("=" * 80)
    
    para_files = [
        f"{DATASETS_DIR}/botriplex_classification/train_para.txt",
        f"{DATASETS_DIR}/botriplex_classification/val_para.txt",
        f"{DATASETS_DIR}/botriplex_classification/test_para.txt",
        f"{DATASETS_DIR}/botriplex/Preprocessed BioTriplex/train_para.txt",
        f"{DATASETS_DIR}/botriplex/Preprocessed BioTriplex/val_para.txt",
        f"{DATASETS_DIR}/botriplex/Preprocessed BioTriplex/test_para.txt",
    ]
    
    all_para_results = []
    all_para_lengths = []
    para_max_overall = 0
    para_max_sample_overall = None
    
    for filepath in para_files:
        res = analyze_file(filepath, tokenizer)
        if res:
            all_para_results.append(res)
            all_para_lengths.extend(res['all_lengths'])
            if res['max'] > para_max_overall:
                para_max_overall = res['max']
                para_max_sample_overall = res['max_sample']
    
    for res in all_para_results:
        print_result(res)
    
    if all_para_lengths:
        all_para_lengths_sorted = sorted(all_para_lengths)
        n = len(all_para_lengths)
        print("\n" + "=" * 80)
        print("📊 GenRel QA 总体统计 (所有 *_para.txt 文件)")
        print("=" * 80)
        print(f"  总样本数: {n}")
        print(f"  Token 长度:")
        print(f"    最小: {min(all_para_lengths)}")
        print(f"    最大: {max(all_para_lengths)} ← 全局最长!")
        print(f"    平均: {sum(all_para_lengths) / n:.1f}")
        print(f"    中位数: {all_para_lengths_sorted[n // 2]}")
        print(f"\n  分位数:")
        for p in [90, 95, 99, 99.5, 99.9]:
            idx = min(int(n * p / 100), n - 1)
            print(f"    P{p}: {all_para_lengths_sorted[idx]}")
        
        # 统计超过各阈值的样本数
        print(f"\n  超过各阈值的样本数:")
        thresholds = [1000, 1500, 2000, 2048, 3000, 4096, 5000, 10000]
        for t in thresholds:
            count = sum(1 for l in all_para_lengths if l > t)
            pct = count / n * 100
            print(f"    > {t}: {count} ({pct:.2f}%)")
    
    # =========================================================================
    # 任务 2: NER (生成任务) - 使用 *_shorter.txt
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("📊 任务 B: NER (生成任务) - *_shorter.txt")
    print("=" * 80)
    
    shorter_files = [
        f"{DATASETS_DIR}/botriplex_generation/train_shorter.txt",
        f"{DATASETS_DIR}/botriplex_generation/val_shorter.txt",
        f"{DATASETS_DIR}/botriplex_generation/test_shorter.txt",
        f"{DATASETS_DIR}/botriplex/Preprocessed BioTriplex/train_shorter.txt",
        f"{DATASETS_DIR}/botriplex/Preprocessed BioTriplex/val_shorter.txt",
        f"{DATASETS_DIR}/botriplex/Preprocessed BioTriplex/test_shorter.txt",
    ]
    
    all_shorter_results = []
    all_shorter_lengths = []
    shorter_max_overall = 0
    shorter_max_sample_overall = None
    
    for filepath in shorter_files:
        res = analyze_file(filepath, tokenizer)
        if res:
            all_shorter_results.append(res)
            all_shorter_lengths.extend(res['all_lengths'])
            if res['max'] > shorter_max_overall:
                shorter_max_overall = res['max']
                shorter_max_sample_overall = res['max_sample']
    
    for res in all_shorter_results:
        print_result(res)
    
    if all_shorter_lengths:
        all_shorter_lengths_sorted = sorted(all_shorter_lengths)
        n = len(all_shorter_lengths)
        print("\n" + "=" * 80)
        print("📊 NER 总体统计 (所有 *_shorter.txt 文件)")
        print("=" * 80)
        print(f"  总样本数: {n}")
        print(f"  Token 长度:")
        print(f"    最小: {min(all_shorter_lengths)}")
        print(f"    最大: {max(all_shorter_lengths)} ← 全局最长!")
        print(f"    平均: {sum(all_shorter_lengths) / n:.1f}")
        print(f"    中位数: {all_shorter_lengths_sorted[n // 2]}")
        print(f"\n  分位数:")
        for p in [90, 95, 99, 99.5, 99.9]:
            idx = min(int(n * p / 100), n - 1)
            print(f"    P{p}: {all_shorter_lengths_sorted[idx]}")
        
        # 统计超过各阈值的样本数
        print(f"\n  超过各阈值的样本数:")
        thresholds = [500, 1000, 1500, 2000, 2048, 3000, 4096, 5000, 10000]
        for t in thresholds:
            count = sum(1 for l in all_shorter_lengths if l > t)
            pct = count / n * 100
            print(f"    > {t}: {count} ({pct:.2f}%)")
    
    # =========================================================================
    # 总结
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("📋 总结: 实际数据 vs max_seq_length=10000")
    print("=" * 80)
    
    if all_para_lengths:
        print(f"\n【GenRel QA 分类任务 (*_para.txt)】")
        print(f"  实际最大 seq_length: {max(all_para_lengths)} tokens")
        print(f"  实际平均 seq_length: {sum(all_para_lengths) / len(all_para_lengths):.1f} tokens")
        print(f"  P95: {sorted(all_para_lengths)[int(len(all_para_lengths) * 0.95)]:.0f} tokens")
        print(f"  P99: {sorted(all_para_lengths)[int(len(all_para_lengths) * 0.99)]:.0f} tokens")
    
    if all_shorter_lengths:
        print(f"\n【NER 生成任务 (*_shorter.txt)】")
        print(f"  实际最大 seq_length: {max(all_shorter_lengths)} tokens")
        print(f"  实际平均 seq_length: {sum(all_shorter_lengths) / len(all_shorter_lengths):.1f} tokens")
        print(f"  P95: {sorted(all_shorter_lengths)[int(len(all_shorter_lengths) * 0.95)]:.0f} tokens")
        print(f"  P99: {sorted(all_shorter_lengths)[int(len(all_shorter_lengths) * 0.99)]:.0f} tokens")
    
    print(f"\n【结论】")
    print(f"  当前设置 max_seq_length = 10000")
    if all_para_lengths and all_shorter_lengths:
        para_max = max(all_para_lengths)
        shorter_max = max(all_shorter_lengths)
        overall_max = max(para_max, shorter_max)
        print(f"  实际数据最大长度: {overall_max} tokens")
        print(f"  设置冗余: {10000 - overall_max} tokens ({(10000/overall_max - 1)*100:.1f}% 过多)")
        
        # 推荐值
        recommended = 2048 if shorter_max <= 2000 else (4096 if shorter_max <= 4000 else 8192)
        print(f"\n  推荐 max_seq_length 设置:")
        print(f"    - {recommended}: 可覆盖 {sum(1 for l in all_shorter_lengths if l <= recommended)/len(all_shorter_lengths)*100:.1f}% 的 NER 样本")
        if all_para_lengths:
            print(f"    - {recommended}: 可覆盖 {sum(1 for l in all_para_lengths if l <= recommended)/len(all_para_lengths)*100:.1f}% 的 QA 样本")

if __name__ == "__main__":
    main()
