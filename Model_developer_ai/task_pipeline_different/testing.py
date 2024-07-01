import sys 
from pathlib import Path
current_file = Path(__file__).resolve()
package_root = current_file.parents[1]
sys.path.append(str(package_root))
import json
import os
import random
import string
import subprocess
from typing import Dict, List, Tuple

import gradio as gr
import yaml

from autotrain.app.models import fetch_models
from autotrain.app.params import get_task_params

# Constants
PREFIX = "autotrain"
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
}

# Create the 'data' directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")


def generate_random_string() -> str:
    """Generates a random string with a prefix."""
    part1 = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    part2 = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    return f"{PREFIX}-{part1}-{part2}"


def _get_params(task: str, param_type: str) -> str:
    """Fetches and formats task parameters."""
    _p = get_task_params(task, param_type=param_type)
    _p["push_to_hub"] = True
    return json.dumps(_p, indent=4)


def on_dataset_change(dataset_source: str) -> Tuple[str, str, str]:
    """Handles changes in dataset source selection."""
    if dataset_source == "Local":
        return "data/", "train", ""
    return "", "", ""


def update_parameters(task: str, param_type: str) -> str:
    """Updates the parameters based on selected task and type."""
    task = TASK_MAP[task]
    param_type = param_type.lower()
    return _get_params(task, param_type)


def update_col_mapping(task: str) -> str:
    """Updates column mapping based on the selected task."""
    task = TASK_MAP[task]
    mapping = {
        "llm:sft": '{"text": "text"}',
        "llm:generic": '{"text": "text"}',
        "llm:dpo": '{"prompt": "prompt", "text": "text", "rejected_text": "rejected_text"}',
        "llm:orpo": '{"prompt": "prompt", "text": "text", "rejected_text": "rejected_text"}',
        "llm:reward": '{"text": "text", "rejected_text": "rejected_text"}',
        "text-classification": '{"text": "text", "label": "target"}',
        "text-regression": '{"text": "text", "label": "target"}',
        "token-classification": '{"text": "tokens", "label": "tags"}',
        "seq2seq": '{"text": "text", "label": "target"}',
        "dreambooth": '{"image": "image"}',
        "image-classification": '{"image": "image", "label": "label"}',
        "image-object-detection": '{"image": "image", "objects": "objects"}',
        "tabular:classification": '{"id": "id", "label": ["target"]}',
        "tabular:regression": '{"id": "id", "label": ["target"]}',
    }
    return mapping.get(task, "Enter column mapping...")


def update_base_model(task: str) -> str:
    """Updates the base model suggestion based on the selected task."""
    task = TASK_MAP[task]
    model_map = {
        "text-classification": MODEL_CHOICES["text-classification"][0],
        "llm:sft": MODEL_CHOICES["llm"][0],
        "llm:orpo": MODEL_CHOICES["llm"][0],
        "llm:generic": MODEL_CHOICES["llm"][0],
        "llm:dpo": MODEL_CHOICES["llm"][0],
        "llm:reward": MODEL_CHOICES["llm"][0],
        "image-classification": MODEL_CHOICES["image-classification"][0],
        "dreambooth": MODEL_CHOICES["dreambooth"][0],
        "seq2seq": MODEL_CHOICES["seq2seq"][0],
        "tabular:classification": MODEL_CHOICES["tabular-classification"][0],
        "tabular:regression": MODEL_CHOICES["tabular-regression"][0],
        "token-classification": MODEL_CHOICES["token-classification"][0],
        "text-regression": MODEL_CHOICES["text-regression"][0],
        "image-object-detection": MODEL_CHOICES["image-object-detection"][0],
    }
    return model_map.get(task, "Enter base model...")


def start_training(
    hf_token: str,
    hf_user: str,
    project_name: str,
    task: str,
    dataset_source: str,
    dataset_path: str,
    train_split: str,
    valid_split: str,
    col_mapping: str,
    base_model: str,
    parameters: str,
) -> None:
    """Starts the training process based on user inputs."""
    try:
        print("Training is starting... Please wait!")
        os.environ["HF_USERNAME"] = hf_user
        os.environ["HF_TOKEN"] = hf_token
        train_split_value = train_split.strip() if train_split.strip() != "" else None
        valid_split_value = valid_split.strip() if valid_split.strip() != "" else None
        params_val = json.loads(parameters)

        if task.startswith("llm"):
            params_val["trainer"] = task.split(":")[1]
            params_val = {k: v for k, v in params_val.items() if k != "trainer"}

        if TASK_MAP[task] == "dreambooth":
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

        if TASK_MAP[task] != "dreambooth":
            config = {
                "task": TASK_MAP[task].split(":")[0],
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
                    "username": "${{HF_USERNAME}}",
                    "token": "${{HF_TOKEN}}",
                    "push_to_hub": push_to_hub,
                },
            }
        else:
            config = {
                "task": TASK_MAP[task],
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
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())

        poll_res = process.poll()
        if poll_res != 0:
            raise Exception(f"Training failed with exit code: {poll_res}")
        print("Training completed successfully!")
    except Exception as e:
        print("An error occurred while starting training!")
        print(f"Error: {e}")
        raise gr.Error(f"Error during training: {e}")


def create_gradio_app() -> gr.Blocks:
    """Creates and configures the Gradio interface."""
    with gr.Blocks() as demo:
        # Hugging Face Credentials
        gr.Markdown("### Hugging Face Credentials")
        hf_token = gr.Textbox(
            label="Hugging Face Write Token",
            type="password",
            value="",
        )
        hf_user = gr.Textbox(
            label="Hugging Face Username",
            value="",
        )

        # Project Details
        gr.Markdown("### Project Details")
        base_model = gr.Textbox(
            label="Base Model",
            value=MODEL_CHOICES["llm"][0],
        )
        project_name = gr.Textbox(
            label="Project Name",
            value=generate_random_string(),
        )
        task_dropdown = gr.Dropdown(
            choices=TASK_NAMES,
            label="Task",
            value=TASK_NAMES[0],
        )

        # Dataset Details
        gr.Markdown("### Dataset Details")
        dataset_source_dropdown = gr.Dropdown(
            choices=["Hugging Face Hub", "Local"],
            label="Source",
            value="Hugging Face Hub",
        )
        dataset_path = gr.Textbox(
            label="Path",
            value="",
        )
        train_split = gr.Textbox(
            label="Train Split",
            value="",
        )
        valid_split = gr.Textbox(
            label="Valid Split",
            value="",
            placeholder="optional",
        )
        col_mapping = gr.Textbox(
            label="Column Mapping",
            value='{"text": "text"}',
        )

        # Parameters
        gr.Markdown("### Parameters")
        parameters_dropdown = gr.Dropdown(
            choices=["Basic", "Full"],
            label="Parameters",
            value="Basic",
        )
        parameters = gr.Code(
            label="Parameters JSON",
            language="json",
            value=_get_params("llm:sft", "basic"),
        )

        # Event listeners for dynamic updates
        dataset_source_dropdown.change(
            on_dataset_change,
            inputs=[dataset_source_dropdown],
            outputs=[dataset_path, train_split, valid_split],
        )
        task_dropdown.change(
            update_col_mapping,
            inputs=[task_dropdown],
            outputs=[col_mapping],
        )
        task_dropdown.change(
            update_parameters,
            inputs=[task_dropdown, parameters_dropdown],
            outputs=[parameters],
        )
        task_dropdown.change(
            update_base_model,
            inputs=[task_dropdown],
            outputs=[base_model],
        )
        parameters_dropdown.change(
            update_parameters,
            inputs=[task_dropdown, parameters_dropdown],
            outputs=[parameters],
        )

        # Training Button
        start_training_button = gr.Button("Start Training")
        start_training_button.click(
            start_training,
            inputs=[
                hf_token,
                hf_user,
                project_name,
                task_dropdown,
                dataset_source_dropdown,
                dataset_path,
                train_split,
                valid_split,
                col_mapping,
                base_model,
                parameters,
            ],
        )

    return demo


# Run the Gradio app
if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch(share=True)