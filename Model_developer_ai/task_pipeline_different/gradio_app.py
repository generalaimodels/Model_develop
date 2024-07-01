import sys
from pathlib import Path
current_file=Path(__file__).resolve()
current_dir=current_file.parents[1]
sys.path.append(str(current_dir))
import gradio as gr
import os
import random
import string
import json
import subprocess
import yaml
from autotrain.app.models import fetch_models
from autotrain.app.params import get_task_params

def generate_random_string():
    prefix = "autotrain"
    part1 = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    part2 = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    return f"{prefix}-{part1}-{part2}"

MODEL_CHOICES = fetch_models()
TASK_NAMES = [
    "LLM SFT",
    "LLM ORPO",
    "LLM Generic",
    "LLM DPO",
    "LLM Reward",
    "Text Classification",
    "Text Regression",
    "Sequence to Sequence",
    "Token Classification",
    "DreamBooth LoRA",
    "Image Classification",
    "Object Detection",
    "Tabular Classification",
    "Tabular Regression",
    "ST Pair",
    "ST Pair Classification",
    "ST Pair Scoring",
    "ST Triplet",
    "ST Question Answering",
]

TASK_MAP = {
    "LLM SFT": "llm:sft",
    "LLM ORPO": "llm:orpo",
    "LLM Generic": "llm:generic",
    "LLM DPO": "llm:dpo",
    "LLM Reward": "llm:reward",
    "Text Classification": "text-classification",
    "Text Regression": "text-regression",
    "Sequence to Sequence": "seq2seq",
    "Token Classification": "token-classification",
    "DreamBooth LoRA": "dreambooth",
    "Image Classification": "image-classification",
    "Object Detection": "image-object-detection",
    "Tabular Classification": "tabular:classification",
    "Tabular Regression": "tabular:regression",
    "ST Pair": "st:pair",
    "ST Pair Classification": "st:pair_class",
    "ST Pair Scoring": "st:pair_score",
    "ST Triplet": "st:triplet",
    "ST Question Answering": "st:qa",
}

def get_params(task, param_type):
    _p = get_task_params(task, param_type=param_type)
    _p["push_to_hub"] = True
    _p = json.dumps(_p, indent=4)
    return _p

def update_parameters(task, param_type):
    task = TASK_MAP[task]
    param_type = param_type.lower()
    return get_params(task, param_type)

def update_col_mapping(task):
    task = TASK_MAP[task]
    if task in ["llm:sft", "llm:generic"]:
        return '{"text": "text"}'
    elif task in ["llm:dpo", "llm:orpo"]:
        return '{"prompt": "prompt", "text": "text", "rejected_text": "rejected_text"}'
    elif task == "llm:reward":
        return '{"text": "text", "rejected_text": "rejected_text"}'
    elif task == "text-classification":
        return '{"text": "text", "label": "target"}'
    elif task == "text-regression":
        return '{"text": "text", "label": "target"}'
    elif task == "token-classification":
        return '{"text": "tokens", "label": "tags"}'
    elif task == "seq2seq":
        return '{"text": "text", "label": "target"}'
    elif task == "dreambooth":
        return '{"image": "image"}'
    elif task == "image-classification":
        return '{"image": "image", "label": "label"}'
    elif task == "image-object-detection":
        return '{"image": "image", "objects": "objects"}'
    elif task == "tabular:classification":
        return '{"id": "id", "label": ["target"]}'
    elif task == "tabular:regression":
        return '{"id": "id", "label": ["target"]}'
    elif task == "st:pair":
        return '{"sentence1": "anchor", "sentence2": "positive"}'
    elif task == "st:pair_class":
        return '{"sentence1": "premise", "sentence2": "hypothesis", "target": "label"}'
    elif task == "st:pair_score":
        return '{"sentence1": "sentence1", "sentence2": "sentence2", "target": "score"}'
    elif task == "st:triplet":
        return '{"sentence1": "anchor", "sentence2": "positive", "sentence3": "negative"}'
    elif task == "st:qa":
        return '{"sentence1": "query", "sentence2": "answer"}'
    else:
        return "Enter column mapping..."

def update_base_model(task):
    if TASK_MAP[task] == "text-classification":
        return MODEL_CHOICES["text-classification"][0]
    elif TASK_MAP[task].startswith("llm"):
        return MODEL_CHOICES["llm"][0]
    elif TASK_MAP[task] == "image-classification":
        return MODEL_CHOICES["image-classification"][0]
    elif TASK_MAP[task] == "dreambooth":
        return MODEL_CHOICES["dreambooth"][0]
    elif TASK_MAP[task] == "seq2seq":
        return MODEL_CHOICES["seq2seq"][0]
    elif TASK_MAP[task] == "tabular:classification":
        return MODEL_CHOICES["tabular-classification"][0]
    elif TASK_MAP[task] == "tabular:regression":
        return MODEL_CHOICES["tabular-regression"][0]
    elif TASK_MAP[task] == "token-classification":
        return MODEL_CHOICES["token-classification"][0]
    elif TASK_MAP[task] == "text-regression":
        return MODEL_CHOICES["text-regression"][0]
    elif TASK_MAP[task] == "image-object-detection":
        return MODEL_CHOICES["image-object-detection"][0]
    elif TASK_MAP[task].startswith("st:"):
        return MODEL_CHOICES["sentence-transformers"][0]
    else:
        return "Enter base model..."

def start_training(hf_user, hf_token, base_model, project_name, task, dataset_source, dataset_path, train_split, valid_split, col_mapping, parameters):
    os.environ["HF_USERNAME"] = hf_user
    os.environ["HF_TOKEN"] = hf_token
    train_split_value = train_split.strip() if train_split.strip() != "" else None
    valid_split_value = valid_split.strip() if valid_split.strip() != "" else None
    params_val = json.loads(parameters)
    task = TASK_MAP[task]

    if task.startswith("llm") or task.startswith("sentence-transformers"):
        params_val["trainer"] = task.split(":")[1]
        params_val = {k: v for k, v in params_val.items() if k != "trainer"}

    if task == "dreambooth":
        prompt = params_val.get("prompt")
        if prompt is None:
            raise ValueError("Prompt is required for DreamBooth task")
        if not isinstance(prompt, str):
            raise ValueError("Prompt should be a string")
        params_val = {k: v for k, v in params_val.items() if k != "prompt"}
    else:
        prompt = None

    push_to_hub = params_val.get("push_to_hub", True)
    if "push_to_hub" in params_val:
        params_val = {k: v for k, v in params_val.items() if k != "push_to_hub"}

    if task != "dreambooth":
        config = {
            "task": task.split(":")[0],
            "base_model": base_model,
            "project_name": project_name,
            "log": "tensorboard",
            "backend": "local",
            "data": {
                "path": dataset_path,
                "train_split": train_split_value,
                "valid_split": valid_split_value,
                "column_mapping": json.loads(col_mapping),
            },
            "params": params_val,
            "hub": {
                "username": "${HF_USERNAME}",
                "token": "${HF_TOKEN}",
                "push_to_hub": push_to_hub,
            },
        }
    else:
        config = {
            "task": task,
            "base_model": base_model,
            "project_name": project_name,
            "backend": "local",
            "data": {
                "path": dataset_path,
                "prompt": prompt,
            },
            "params": params_val,
            "hub": {
                "username": "${HF_USERNAME}",
                "token": "${HF_TOKEN}",
                "push_to_hub": push_to_hub,
            },
        }

    with open("config.yml", "w") as f:
        yaml.dump(config, f)

    cmd = "autotrain --config config.yml"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = []
    while True:
        line = process.stdout.readline()
        if line == "" and process.poll() is not None:
            break
        if line:
            output.append(line.strip())
    return "\n".join(output)

# Gradio Interface
def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# AutoTrain Project Setup")

        with gr.Row():
            hf_user = gr.Textbox(label="Hugging Face Username")
            hf_token = gr.Textbox(label="Hugging Face Token", type="password")

        with gr.Row():
            base_model = gr.Textbox(label="Base Model", value=MODEL_CHOICES["text-classification"][0])
            project_name = gr.Textbox(label="Project Name", value=generate_random_string())

        with gr.Row():
            task = gr.Dropdown(label="Task", choices=TASK_NAMES, value="Text Classification")
            dataset_source = gr.Radio(label="Dataset Source", choices=["Hub", "Local"], value="Hub")

        dataset_path = gr.Textbox(label="Dataset Path")
        train_split = gr.Textbox(label="Train Split", value="train")
        valid_split = gr.Textbox(label="Valid Split", value="valid")

        col_mapping = gr.Textbox(label="Column Mapping", value='{"text": "text", "label": "target"}')
        parameters = gr.Textbox(label="Parameters", value=json.dumps(get_params("text-classification", "required"), indent=4), lines=5)

        task.change(fn=update_col_mapping, inputs=task, outputs=col_mapping)
        task.change(fn=update_parameters, inputs=[task, gr.Dropdown(choices=["Required", "Optional"])], outputs=parameters)
        task.change(fn=update_base_model, inputs=task, outputs=base_model)

        train_button = gr.Button("Start Training")

        output = gr.Textbox(label="Output", lines=10)

        train_button.click(fn=start_training, 
                           inputs=[hf_user, hf_token, base_model, project_name, task, dataset_source, dataset_path, train_split, valid_split, col_mapping, parameters], 
                           outputs=output)

    return demo

demo = create_ui()
demo.launch(share=True)
