import os
from pathlib import Path
from typing import Union
import pandas as pd
import yaml
from huggingface_hub import snapshot_download


def clone_repo(repo_url: str, local_dir: Union[str, Path]) -> None:
    """Clone the repository."""
    os.system(f"git clone {repo_url} {local_dir}")


def install_requirements(requirements_file: Union[str, Path]) -> None:
    """Install the required dependencies."""
    os.system(f"pip install -r {requirements_file}")


def download_model(model_url: str, local_dir: Union[str, Path]) -> None:
    """Download the model."""
    os.system(f"wget {model_url}")
    os.system(f"tar -xf {model_url.split('/')[-1]} -C {local_dir}")


def download_model_from_hf(model_id: str, local_dir: Union[str, Path]) -> None:
    """Download the model from Hugging Face."""
    snapshot_download(
        repo_id=model_id,
        allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"],
        local_dir=local_dir,
    )


def read_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Read data from a URL or local file path."""
    if file_path.startswith("http"):
        return pd.read_parquet(file_path)
    else:
        file_extension = Path(file_path).suffix
        if file_extension == ".json":
            return pd.read_json(file_path)
        elif file_extension == ".csv":
            return pd.read_csv(file_path)
        elif file_extension == ".parquet":
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")


def save_data(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """Save data to a JSONL file."""
    df.to_json(file_path, orient="records", lines=True)


def reformat_data(data_file: Union[str, Path]) -> None:
    """Reformat the data into the correct format."""
    os.system(f"python -m utils.reformat_data {data_file}")


def create_config(config_data: dict, file_path: Union[str, Path]) -> None:
    """Create a YAML configuration file."""
    with open(file_path, "w") as file:
        yaml.dump(config_data, file)


def main() -> None:
    # Clone the repository
    repo_url = "https://github.com/mistralai/mistral-finetune.git"
    local_dir = "mistral-finetune"
    clone_repo(repo_url, local_dir)

    # Install requirements
    requirements_file = Path(local_dir) / "requirements.txt"
    install_requirements(requirements_file)

    # Download the model
    model_url = "https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-v0.3.tar"
    model_dir = Path("mistral_models")
    model_dir.mkdir(exist_ok=True)
    download_model(model_url, model_dir)

    # Read data
    file_path = input("Enter the URL or path to the data file (JSON, CSV, or Parquet): ")
    df = read_data(file_path)

    # Split data into train and eval
    train_ratio = 0.8
    train_size = int(len(df) * train_ratio)
    df_train, df_eval = df[:train_size], df[train_size:]

    # Save data to JSONL files
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    save_data(df_train, data_dir / "train.jsonl")
    save_data(df_eval, data_dir / "eval.jsonl")

    # Reformat data
    reformat_data(data_dir / "train.jsonl")
    reformat_data(data_dir / "eval.jsonl")

    # Set environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Create configuration
    config_data = {
        "data": {
            "instruct_data": str(data_dir / "train.jsonl"),
            "data": "",
            "eval_instruct_data": str(data_dir / "eval.jsonl"),
        },
        "model_id_or_path": str(model_dir),
        "lora": {"rank": 64},
        "seq_len": 8192,
        "batch_size": 1,
        "num_microbatches": 8,
        "max_steps": 100,
        "optim": {
            "lr": 1e-4,
            "weight_decay": 0.1,
            "pct_start": 0.05,
        },
        "seed": 0,
        "log_freq": 1,
        "eval_freq": 100,
        "no_eval": False,
        "ckpt_freq": 100,
        "save_adapters": True,
        "run_dir": "test_ultra",
    }
    create_config(config_data, "config.yaml")

    # Start training
    os.system("torchrun --nproc-per-node 1 -m train config.yaml")


if __name__ == "__main__":
    main()