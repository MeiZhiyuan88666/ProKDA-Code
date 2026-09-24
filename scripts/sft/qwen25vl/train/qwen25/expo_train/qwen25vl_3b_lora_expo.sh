#!/bin/bash
set -e

model_name=qwen25vl_3b
dataset_name=FB
export WANDB_PROJECT="LLAMAFACTORY_hateful_qwen25vl"
export WANDB_RUN_GROUP="Finetuning_${dataset_name}_${model_name}_sft"

export WANDB_NAME=${dataset_name}_${model_name}_sft_expo_wiki_epoch
CUDA_VISIBLE_DEVICES=3 llamafactory-cli train scripts/sft/qwen25vl/qwen25vl_3b_lora_hmc_expo_wiki.yaml