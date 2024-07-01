import sys
from pathlib import Path
file=Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
from FAST_ANALYSIS import AiModelForHemanth
import logging
from typing import Any, Dict, Optional, List
import torch
from torch.nn import Module
from dataclasses import dataclass
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    save_path: str
    load_path: Optional[str] = None
    num_shards: int = 1

class ModelManager:
    def __init__(self, model: Module, config: ModelConfig):
        self.model = model
        self.config = config
        self.load_path = config.load_path or config.save_path

    def save_model(self) -> None:
        """Save the model weights in shards."""
        try:
            os.makedirs(self.config.save_path, exist_ok=True)
            shards = self.split_model()
            for i, shard in enumerate(shards):
                shard_path = os.path.join(self.config.save_path, f"shard_{i}.pt")
                torch.save(shard, shard_path)
            logger.info(f"Model saved successfully in {self.config.num_shards} shards.")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self) -> None:
        """Load the model weights from shards."""
        try:
            shards = []
            for i in range(self.config.num_shards):
                shard_path = os.path.join(self.load_path, f"shard_{i}.pt")
                shard = torch.load(shard_path)
                shards.append(shard)
            merged_state_dict = self.merge_shards(shards)
            self.model.load_state_dict(merged_state_dict)
            logger.info("Model loaded successfully from shards.")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def get_model_architecture(self) -> str:
        """Get the model architecture as a string."""
        return str(self.model)

    def get_model_config(self) -> Dict[str, Any]:
        """Get the model configuration."""
        return self.model.config.to_dict() if hasattr(self.model, 'config') else {}

    def split_model(self) -> List[Dict[str, torch.Tensor]]:
        """Split the model into shards."""
        try:
            shards = []
            state_dict = self.model.state_dict()
            keys = list(state_dict.keys())
            shard_size = len(keys) // self.config.num_shards
            
            for i in range(self.config.num_shards):
                start = i * shard_size
                end = (i + 1) * shard_size if i < self.config.num_shards - 1 else len(keys)
                shard = {k: state_dict[k] for k in keys[start:end]}
                shards.append(shard)
            
            logger.info(f"Model split into {self.config.num_shards} shards.")
            return shards
        except Exception as e:
            logger.error(f"Error splitting model: {str(e)}")
            raise

    @staticmethod
    def merge_shards(shards: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Merge model shards back into a single state dict."""
        try:
            merged_state_dict = {}
            for shard in shards:
                merged_state_dict.update(shard)
            
            logger.info("Model shards merged successfully.")
            return merged_state_dict
        except Exception as e:
            logger.error(f"Error merging model shards: {str(e)}")
            raise
import logging
from typing import Any, Dict, Optional, List
import torch
from torch.nn import Module
from dataclasses import dataclass
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    save_path: str
    load_path: Optional[str] = None
    num_shards: int = 1

class ModelManager_test:
    def __init__(self, model: Module, config: ModelConfig):
        self.model = model
        self.config = config
        self.load_path = config.load_path or config.save_path

    def save_model(self) -> None:
        """Save the model weights in shards."""
        try:
            os.makedirs(self.config.save_path, exist_ok=True)
            shards = self.split_model()
            for i, shard in enumerate(shards):
                shard_path = os.path.join(self.config.save_path, f"shard_{i}.pt")
                torch.save(shard, shard_path)
            logger.info(f"Model saved successfully in {self.config.num_shards} shards.")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self) -> None:
        """Load the model weights from shards."""
        try:
            shards = []
            for i in range(self.config.num_shards):
                shard_path = os.path.join(self.load_path, f"shard_{i}.pt")
                shard = torch.load(shard_path)
                shards.append(shard)
            merged_state_dict = self.merge_shards(shards)
            self.model.load_state_dict(merged_state_dict)
            logger.info("Model loaded successfully from shards.")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def get_model_architecture(self) -> str:
        """Get the model architecture as a string."""
        return str(self.model)

    def get_model_config(self) -> Dict[str, Any]:
        """Get the model configuration."""
        return self.model.config.to_dict() if hasattr(self.model, 'config') else {}

    def split_model(self) -> List[Dict[str, torch.Tensor]]:
        """Split the model into shards."""
        try:
            shards = []
            state_dict = self.model.state_dict()
            keys = list(state_dict.keys())
            shard_size = len(keys) // self.config.num_shards
            
            for i in range(self.config.num_shards):
                start = i * shard_size
                end = (i + 1) * shard_size if i < self.config.num_shards - 1 else len(keys)
                shard = {k: state_dict[k] for k in keys[start:end]}
                shards.append(shard)
            
            logger.info(f"Model split into {self.config.num_shards} shards.")
            return shards
        except Exception as e:
            logger.error(f"Error splitting model: {str(e)}")
            raise

    @staticmethod
    def merge_shards(shards: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Merge model shards back into a single state dict."""
        try:
            merged_state_dict = {}
            for shard in shards:
                merged_state_dict.update(shard)
            
            logger.info("Model shards merged successfully.")
            return merged_state_dict
        except Exception as e:
            logger.error(f"Error merging model shards: {str(e)}")
            raise


def main():
    model=AiModelForHemanth.load_model(
    model_type="causal_lm",
    model_name_or_path="gpt2",
    cache_dir=r"E:\LLMS\Fine-tuning\data")


    # Initialize ModelManager
    config = ModelConfig(save_path=r"E:\LLMS\Fine-tuning\data\model_checkpoint1.pth",
                         load_path=r"E:\LLMS\Fine-tuning\data\model_checkpoint2.pth",
                         num_shards=2
                         )
    manager = ModelManager(model, config)

    # Save the model
    manager.save_model()

    # Load the model
    manager.load_model()

    # Get model architecture
    architecture = manager.get_model_architecture()
    logger.info(f"Model architecture: {architecture}")

    # # Split the model
    # shards = manager.split_model()
    # logger.info(f"Number of shards: {len(shards)}")

    # # Merge the shards
    # merged_model = ModelManager.merge_shards(shards)
    # logger.info(f"Merged model: {merged_model}")

if __name__ == "__main__":
    main()