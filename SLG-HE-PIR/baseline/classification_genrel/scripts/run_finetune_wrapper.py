#!/usr/bin/env python3
"""Wrapper script to run finetuning with correct parameters."""

import sys
import runpy
import os
import argparse

# Parse additional arguments for resume/epochs
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--timestamp', type=str, required=True)
parser.add_argument('--from_peft_checkpoint', type=str, default='')
parser.add_argument('--log_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
args, unknown = parser.parse_known_args()

# Setup paths
sys.path.insert(0, '/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/baseline/llama-rec/src')
sys.path.insert(0, '/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/baseline/llama-rec/_compat')

# Import compat patch
import transformers_59_patch

# Build sys.argv for fire
sys.argv = [
    'finetuning.py',
    '--use_peft',
    '--peft_method', 'lora',
    '--r', '8',
    '--lora_alpha', '16',
    '--lora_dropout', '0.05',
    '--model_name', '/root/autodl-tmp/CipherForgeCode/hf_cache/Llama-3-1-8B-I',
    '--output_dir', args.output_dir,
    '--batch_size_training', '1',
    '--batching_strategy', 'padding',
    '--num_epochs', '1',
    '--lr', '1e-4',
    '--weight_decay', '0.0',
    '--dataset', 'biotriplex_qakshot_dataset',
    '--num_of_shots', '0',
    '--context_length', '10000',
    '--data_path', '/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/datasets/botriplex/Preprocessed BioTriplex/',
    '--use_entity_tokens_as_targets', 'False',
    '--entity_special_tokens', 'False',
    '--use_fast_kernels', 'True',
    '--upweight_minority_class', 'False',
    '--bidirectional_attention_in_entity_tokens', 'False',
    '--enable_fsdp', 'False',
    '--return_neg_relations', 'False',
    '--use_wandb', 'False',
    '--general_relations', 'True',
    '--run_validation', 'True',
    '--save_model', 'True',
    '--save_metrics', 'True',
]

# Add from_peft_checkpoint if provided
if args.from_peft_checkpoint:
    sys.argv.extend(['--from_peft_checkpoint', args.from_peft_checkpoint])

# Pass through any unknown args (like dataset-specific params)
sys.argv.extend(unknown)

print(f"[Epoch {args.epoch}] Starting finetuning with args:", sys.argv[1:])

# Run the finetuning script
runpy.run_path('/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/baseline/llama-rec/recipes/quickstart/finetuning/finetuning.py', run_name='__main__')
