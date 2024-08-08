#!/bin/bash

# Set default values
NUM_NODES=1
NUM_GPUS=1
AUTOTUNING_MODE="run"
MODEL_NAME="gpt2"
DS_CONFIG="ds_config.json"
HF_SCRIPT="run_clm.py"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --nodes NUM         Number of nodes (default: $NUM_NODES)"
    echo "  --gpus NUM          Number of GPUs per node (default: $NUM_GPUS)"
    echo "  --autotuning MODE   Autotuning mode: run or tune (default: $AUTOTUNING_MODE)"
    echo "  --model NAME        Model name or path (default: $MODEL_NAME)"
    echo "  --config FILE       DeepSpeed config file (default: $DS_CONFIG)"
    echo "  --script FILE       Hugging Face script (default: $HF_SCRIPT)"
    echo "  --help              Display this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nodes) NUM_NODES="$2"; shift 2 ;;
        --gpus) NUM_GPUS="$2"; shift 2 ;;
        --autotuning) AUTOTUNING_MODE="$2"; shift 2 ;;
        --model) MODEL_NAME="$2"; shift 2 ;;
        --config) DS_CONFIG="$2"; shift 2 ;;
        --script) HF_SCRIPT="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Validate autotuning mode
if [[ "$AUTOTUNING_MODE" != "run" && "$AUTOTUNING_MODE" != "tune" ]]; then
    echo "Error: Invalid autotuning mode. Use 'run' or 'tune'."
    exit 1
fi

# Ensure DeepSpeed config file exists
if [[ ! -f "$DS_CONFIG" ]]; then
    echo "Error: DeepSpeed config file '$DS_CONFIG' not found."
    exit 1
fi

# Ensure Hugging Face script exists
if [[ ! -f "$HF_SCRIPT" ]]; then
    echo "Error: Hugging Face script '$HF_SCRIPT' not found."
    exit 1
fi

# Function to run DeepSpeed with autotuning
run_deepspeed() {
    deepspeed --autotuning="$AUTOTUNING_MODE" \
              --num_nodes="$NUM_NODES" \
              --num_gpus="$NUM_GPUS" \
              "$HF_SCRIPT" \
              --deepspeed "$DS_CONFIG" \
              --model_name_or_path "$MODEL_NAME" \
              --do_train \
              --do_eval \
              --fp16 \
              --per_device_train_batch_size 8 \
              --gradient_accumulation_steps 1 \
              "$@"
}

# Execute DeepSpeed with autotuning
run_deepspeed "$@"