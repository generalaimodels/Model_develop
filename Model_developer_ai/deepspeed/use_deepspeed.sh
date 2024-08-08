#!/bin/bash

# Set default values
NUM_NODES=1
NUM_GPUS=1
MODEL_NAME="gpt2"
MAX_TRAIN_BATCH_SIZE=1024
AUTOTUNING_MODE="run"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --num_nodes)
        NUM_NODES="$2"
        shift
        shift
        ;;
        --num_gpus)
        NUM_GPUS="$2"
        shift
        shift
        ;;
        --model_name)
        MODEL_NAME="$2"
        shift
        shift
        ;;
        --max_train_batch_size)
        MAX_TRAIN_BATCH_SIZE="$2"
        shift
        shift
        ;;
        --autotuning_mode)
        AUTOTUNING_MODE="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Create DeepSpeed configuration file
cat << EOF > ds_config.json
{
    "autotuning": {
        "enabled": true,
        "max_train_batch_size": $MAX_TRAIN_BATCH_SIZE,
        "arg_mappings": {
            "train_micro_batch_size_per_gpu": "--per_device_train_batch_size",
            "gradient_accumulation_steps": "--gradient_accumulation_steps"
        }
    },
    "zero_optimization": {
        "stage": [0, 1, 2, 3]
    },
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "fp16": {
        "enabled": true
    }
}
EOF

# Run DeepSpeed with Autotuning
deepspeed --autotuning=$AUTOTUNING_MODE \
    --num_nodes=$NUM_NODES \
    --num_gpus=$NUM_GPUS \
    $HF_PATH/transformers/examples/pytorch/language-modeling/run_clm.py \
    --deepspeed ds_config.json \
    --model_name_or_path $MODEL_NAME \
    --do_train \
    --do_eval \
    --fp16 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --output_dir ./output \
    --overwrite_output_dir

# Clean up
rm ds_config.json