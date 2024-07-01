import copy
from typing import Any, Dict, Union, Tuple
import torch
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def replace_weights_with_noise(
    model: Union[Dict[str, Any], Any],
    target_key: str,
    noise_scale: float = 0.1
) -> Tuple[Union[Dict[str, Any], Any], Union[torch.Tensor, np.ndarray]]:
    """
    Replace weights of a specific tensor in the model with random noise.

    Args:
        model: The input model (dictionary of dictionaries or any structure).
        target_key: The key of the tensor to be replaced with noise.
        noise_scale: Scale of the random noise (default: 0.1).

    Returns:
        A tuple containing:
        - The modified model with the specified tensor replaced by random noise.
        - The original weights that were replaced.

    Raises:
        KeyError: If the target_key is not found in the model.
        ValueError: If the target tensor is not a torch.Tensor or numpy.ndarray.
    """
    start_time = time.time()

    try:
        # Create a deep copy of the model to avoid modifying the original
        modified_model = copy.deepcopy(model)

        # Traverse the model structure to find the target tensor
        target_tensor = _get_nested_item(modified_model, target_key)
        original_weights = copy.deepcopy(target_tensor)

        if isinstance(target_tensor, torch.Tensor):
            # Generate random noise with the same shape, dtype, and device as the target tensor
            noise = torch.randn_like(target_tensor) * noise_scale
            # Replace the target tensor with noise
            _set_nested_item(modified_model, target_key, noise)
        elif isinstance(target_tensor, np.ndarray):
            # Generate random noise with the same shape and dtype as the target array
            noise = np.random.randn(*target_tensor.shape).astype(target_tensor.dtype) * noise_scale
            # Replace the target array with noise
            _set_nested_item(modified_model, target_key, noise)
        else:
            raise ValueError("Target tensor must be a torch.Tensor or numpy.ndarray")

        end_time = time.time()
        logger.info(f"Replaced weights with noise in {end_time - start_time:.4f} seconds")

        return modified_model, original_weights

    except KeyError:
        logger.error(f"Target key '{target_key}' not found in the model structure")
        raise
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


def _get_nested_item(obj: Any, key: str) -> Any:
    """
    Retrieve a nested item from a dictionary or object structure.

    Args:
        obj: The input object or dictionary.
        key: The key to retrieve, using dot notation for nested structures.

    Returns:
        The value associated with the given key.

    Raises:
        KeyError: If the key is not found in the structure.
    """
    keys = key.split('.')
    for k in keys:
        if isinstance(obj, dict):
            obj = obj[k]
        else:
            obj = getattr(obj, k)
    return obj


def _set_nested_item(obj: Any, key: str, value: Any) -> None:
    """
    Set a nested item in a dictionary or object structure.

    Args:
        obj: The input object or dictionary.
        key: The key to set, using dot notation for nested structures.
        value: The value to set for the given key.

    Raises:
        KeyError: If the key is not found in the structure.
        AttributeError: If the object does not have the specified attribute.
    """
    keys = key.split('.')
    for k in keys[:-1]:
        if isinstance(obj, dict):
            obj = obj[k]
        else:
            obj = getattr(obj, k)
    
    if isinstance(obj, dict):
        obj[keys[-1]] = value
    else:
        setattr(obj, keys[-1], value)


def restore_original_weights(
    model: Union[Dict[str, Any], Any],
    target_key: str,
    original_weights: Union[torch.Tensor, np.ndarray]
) -> Union[Dict[str, Any], Any]:
    """
    Restore the original weights of a specific tensor in the model.

    Args:
        model: The input model (dictionary of dictionaries or any structure).
        target_key: The key of the tensor to be restored.
        original_weights: The original weights to be restored.

    Returns:
        The model with the specified tensor restored to its original weights.

    Raises:
        KeyError: If the target_key is not found in the model.
        ValueError: If the original_weights type doesn't match the target tensor type.
    """
    try:
        # Create a deep copy of the model to avoid modifying the original
        restored_model = copy.deepcopy(model)

        # Get the current tensor at the target key
        current_tensor = _get_nested_item(restored_model, target_key)

        if not isinstance(original_weights, type(current_tensor)):
            raise ValueError("Original weights type doesn't match the target tensor type")

        # Restore the original weights
        _set_nested_item(restored_model, target_key, original_weights)

        logger.info(f"Restored original weights for key '{target_key}'")
        return restored_model

    except KeyError:
        logger.error(f"Target key '{target_key}' not found in the model structure")
        raise
    except Exception as e:
        logger.error(f"An error occurred while restoring weights: {str(e)}")
        raise

import copy
import logging
from typing import Any, Dict, Union, Tuple, Optional

import numpy as np
import torch
from torch import nn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelWeightManager:
    """A class to manage model weights, including noise injection and restoration."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the ModelWeightManager.

        Args:
            seed: Optional seed for random number generation.
        """
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def replace_weights_with_noise(
        self,
        model: Union[nn.Module, Dict[str, Any]],
        target_key: str,
        noise_scale: float = 0.1,
        device: Optional[torch.device] = None
    ) -> Tuple[Union[nn.Module, Dict[str, Any]], Union[torch.Tensor, np.ndarray]]:
        """
        Replace weights of a specific tensor in the model with random noise.

        Args:
            model: The input model (nn.Module or dictionary structure).
            target_key: The key of the tensor to be replaced with noise.
            noise_scale: Scale of the random noise (default: 0.1).
            device: The device to use for tensor operations (default: None, uses current device).

        Returns:
            A tuple containing:
            - The modified model with the specified tensor replaced by random noise.
            - The original weights that were replaced.

        Raises:
            KeyError: If the target_key is not found in the model.
            ValueError: If the target tensor is not a torch.Tensor or numpy.ndarray.
        """
        try:
            modified_model = copy.deepcopy(model)
            target_tensor = self._get_nested_item(modified_model, target_key)
            original_weights = copy.deepcopy(target_tensor)

            if isinstance(target_tensor, torch.Tensor):
                noise = torch.randn_like(target_tensor, device=device) * noise_scale
                self._set_nested_item(modified_model, target_key, noise)
            elif isinstance(target_tensor, np.ndarray):
                noise = np.random.randn(*target_tensor.shape).astype(target_tensor.dtype) * noise_scale
                self._set_nested_item(modified_model, target_key, noise)
            else:
                raise ValueError("Target tensor must be a torch.Tensor or numpy.ndarray")

            logger.info(f"Replaced weights with noise for key '{target_key}'")
            return modified_model, original_weights

        except KeyError:
            logger.error(f"Target key '{target_key}' not found in the model structure")
            raise
        except Exception as e:
            logger.error(f"An error occurred while replacing weights: {str(e)}")
            raise

    def restore_original_weights(
        self,
        model: Union[nn.Module, Dict[str, Any]],
        target_key: str,
        original_weights: Union[torch.Tensor, np.ndarray]
    ) -> Union[nn.Module, Dict[str, Any]]:
        """
        Restore the original weights of a specific tensor in the model.

        Args:
            model: The input model (nn.Module or dictionary structure).
            target_key: The key of the tensor to be restored.
            original_weights: The original weights to be restored.

        Returns:
            The model with the specified tensor restored to its original weights.

        Raises:
            KeyError: If the target_key is not found in the model.
            ValueError: If the original_weights type doesn't match the target tensor type.
        """
        try:
            restored_model = copy.deepcopy(model)
            current_tensor = self._get_nested_item(restored_model, target_key)

            if not isinstance(original_weights, type(current_tensor)):
                raise ValueError("Original weights type doesn't match the target tensor type")

            self._set_nested_item(restored_model, target_key, original_weights)
            logger.info(f"Restored original weights for key '{target_key}'")
            return restored_model

        except KeyError:
            logger.error(f"Target key '{target_key}' not found in the model structure")
            raise
        except Exception as e:
            logger.error(f"An error occurred while restoring weights: {str(e)}")
            raise

    @staticmethod
    def _get_nested_item(obj: Any, key: str) -> Any:
        """Retrieve a nested item from a dictionary or object structure."""
        keys = key.split('.')
        for k in keys:
            if isinstance(obj, dict):
                obj = obj[k]
            elif isinstance(obj, nn.Module):
                obj = getattr(obj, k)
            else:
                raise ValueError(f"Unsupported object type: {type(obj)}")
        return obj

    @staticmethod
    def _set_nested_item(obj: Any, key: str, value: Any) -> None:
        """Set a nested item in a dictionary or object structure."""
        keys = key.split('.')
        for k in keys[:-1]:
            if isinstance(obj, dict):
                obj = obj[k]
            elif isinstance(obj, nn.Module):
                obj = getattr(obj, k)
            else:
                raise ValueError(f"Unsupported object type: {type(obj)}")
        
        if isinstance(obj, dict):
            obj[keys[-1]] = value
        elif isinstance(obj, nn.Module):
            if isinstance(getattr(obj, keys[-1]), nn.Parameter):
                setattr(obj, keys[-1], nn.Parameter(value))
            else:
                setattr(obj, keys[-1], value)
        else:
            raise ValueError(f"Unsupported object type: {type(obj)}")

def main():
    # Example usage
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )

    weight_manager = ModelWeightManager(seed=42)

    try:
        # Task 1: Replace weights with noise
        modified_model, original_weights = weight_manager.replace_weights_with_noise(
            model, "0.weight", noise_scale=0.1, device=torch.device("cpu")
        )
        logger.info("Weights replaced with noise")

        # Task 2: Restore original weights
        restored_model = weight_manager.restore_original_weights(
            modified_model, "0.weight", original_weights
        )
        logger.info("Original weights restored")

        # Verify that the restored weights match the original weights
        assert torch.all(torch.eq(model[0].weight, restored_model[0].weight))
        logger.info("Restoration successful: Restored weights match the original weights")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()