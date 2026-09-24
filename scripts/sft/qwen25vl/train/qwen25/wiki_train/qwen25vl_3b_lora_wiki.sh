#!/bin/bash
set -e
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/local/cuda/lib64"

model_name=qwen25vl_3b
dataset_name=FB
export WANDB_PROJECT="LLAMAFACTORY_hateful_qwen25vl"
export WANDB_RUN_GROUP="Finetuning_${dataset_name}_${model_name}_sft"


export WANDB_NAME=${dataset_name}_${model_name}__sft_wiki_epoch1
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train qwen25vl_3b_lora_mami_back_epoch1.yaml
