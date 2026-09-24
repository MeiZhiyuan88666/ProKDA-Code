#!/bin/bash
set -e

model_name=qwen25vl_3b
dataset_name=FB
export WANDB_PROJECT="LLAMAFACTORY_hateful_qwen25vl_3b"
export WANDB_RUN_GROUP="Finetuning_${dataset_name}_${model_name}_sft"

export WANDB_NAME=${dataset_name}_${model_name}_sft_dpo_hmc_expo_low_conf
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=1,3 llamafactory-cli train qwen25vl_3b_sft_dpo_hmc_expo_low_conf.yaml