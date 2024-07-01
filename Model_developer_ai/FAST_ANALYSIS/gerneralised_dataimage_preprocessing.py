import logging
import numpy as np
import torch
from typing import Any, Dict, Tuple, Optional
import albumentations as A
from sklearn import metrics


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageClassificationDataset:
    def __init__(self, data: Any, transforms: A.Compose, config: Any):
        self.data = data
        self.transforms = transforms
        self.config = config

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        image = self.data[item][self.config.image_column]
        target = int(self.data[item][self.config.target_column])

        # Transform image
        image = self.transforms(image=np.array(image.convert("RGB")))["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "pixel_values": torch.tensor(image, dtype=torch.float),
            "labels": torch.tensor(target, dtype=torch.long),
        }



def _binary_classification_metrics(pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    raw_predictions, labels = pred
    predictions = np.argmax(raw_predictions, axis=1)
    result = {
        "f1": metrics.f1_score(labels, predictions),
        "precision": metrics.precision_score(labels, predictions),
        "recall": metrics.recall_score(labels, predictions),
        "auc": metrics.roc_auc_score(labels, raw_predictions[:, 1]),
        "accuracy": metrics.accuracy_score(labels, predictions),
    }
    return result

def _multi_class_classification_metrics(pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    raw_predictions, labels = pred
    predictions = np.argmax(raw_predictions, axis=1)
    results = {
        "f1_macro": metrics.f1_score(labels, predictions, average="macro"),
        "f1_micro": metrics.f1_score(labels, predictions, average="micro"),
        "f1_weighted": metrics.f1_score(labels, predictions, average="weighted"),
        "precision_macro": metrics.precision_score(labels, predictions, average="macro"),
        "precision_micro": metrics.precision_score(labels, predictions, average="micro"),
        "precision_weighted": metrics.precision_score(labels, predictions, average="weighted"),
        "recall_macro": metrics.recall_score(labels, predictions, average="macro"),
        "recall_micro": metrics.recall_score(labels, predictions, average="micro"),
        "recall_weighted": metrics.recall_score(labels, predictions, average="weighted"),
        "accuracy": metrics.accuracy_score(labels, predictions),
    }
    return results


def process_data(train_data: Any, valid_data: Optional[Any], image_processor: Any, config: Any) -> Tuple[ImageClassificationDataset, Optional[ImageClassificationDataset]]:
    try:
        if "shortest_edge" in image_processor.size:
            size = image_processor.size["shortest_edge"]
        else:
            size = (image_processor.size["height"], image_processor.size["width"])
        height, width = size if isinstance(size, tuple) else (size, size)
    except Exception as e:
        logger.error("Error in processing image size: %s", e)
        raise

    train_transforms = A.Compose(
        [
            A.RandomResizedCrop(height=height, width=width),
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ]
    )

    train_dataset = ImageClassificationDataset(train_data, train_transforms, config)
    valid_dataset = ImageClassificationDataset(valid_data, val_transforms, config) if valid_data else None

    return train_dataset, valid_dataset






