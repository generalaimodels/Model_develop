import copy
from typing import Any, Dict, Union
import torch
import numpy as np




def replace_weights_with_noise(
    model: Union[Dict[str, Any], Any],
    target_key: str,
    noise_scale: float = 0.1
) -> Union[Dict[str, Any], Any]:
    """
    Replace weights of a specific tensor in the model with random noise.

    Args:
        model: The input model (dictionary of dictionaries or any structure).
        target_key: The key of the tensor to be replaced with noise.
        noise_scale: Scale of the random noise (default: 0.1).

    Returns:
        The modified model with the specified tensor replaced by random noise.

    Raises:
        KeyError: If the target_key is not found in the model.
        ValueError: If the target tensor is not a torch.Tensor or numpy.ndarray.
    """
    try:
        # Create a deep copy of the model to avoid modifying the original
        modified_model = copy.deepcopy(model)

        # Traverse the model structure to find the target tensor
        target_tensor = _get_nested_item(modified_model, target_key)

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

        return modified_model

    except KeyError:
        raise KeyError(f"Target key '{target_key}' not found in the model structure")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}")


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



def replace_weights_with_noise(
    model: Union[Dict[str, Any], Any],
    target_key: str,
    seed: int = 42
) -> Union[Dict[str, Any], Any]:
    """
    Replace weights of a specific tensor in the model with random noise.

    Args:
        model: The input model (dictionary of dictionaries or any structure).
        target_key: The key of the tensor to be replaced.
        seed: Seed for random number generation (default: 42).

    Returns:
        The modified model with the specified tensor replaced by random noise.

    Raises:
        KeyError: If the target_key is not found in the model.
        ValueError: If the target tensor is not a PyTorch tensor or NumPy array.
    """
    try:
        # Create a deep copy of the model to avoid modifying the original
        modified_model = copy.deepcopy(model)

        # Set the random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        def replace_tensor_recursive(obj: Any, key: str) -> Any:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == key:
                        if isinstance(v, torch.Tensor):
                            return torch.randn_like(v)
                        elif isinstance(v, np.ndarray):
                            return np.random.randn(*v.shape).astype(v.dtype)
                        else:
                            raise ValueError(f"Unsupported tensor type: {type(v)}")
                    obj[k] = replace_tensor_recursive(v, key)
            elif isinstance(obj, list):
                return [replace_tensor_recursive(item, key) for item in obj]
            return obj

        # Recursively search and replace the target tensor
        modified_model = replace_tensor_recursive(modified_model, target_key)

        return modified_model

    except KeyError as e:
        raise KeyError(f"Target key '{target_key}' not found in the model.") from e
    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}") from e


# Example usage
if __name__ == "__main__":
    # Sample model structure
    model = {
        "layer1": {
            "weights": torch.randn(3, 3),
            "bias": torch.randn(3)
        },
        "layer2": {
            "weights": np.random.randn(4, 4),
            "bias": np.random.randn(4)
        }
    }

    try:
        # Replace weights of layer1 with random noise
        modified_model = replace_weights_with_noise(model, "weights")
        print("Modified model:", modified_model)
    except (KeyError, ValueError, RuntimeError) as e:
        print(f"Error: {str(e)}")