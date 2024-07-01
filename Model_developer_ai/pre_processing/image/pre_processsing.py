import sys
from pathlib import Path
file=Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
import albumentations as A
import torch
import numpy as np
import logging
from albumentations.pytorch import ToTensorV2
from FAST_ANALYSIS import AdvancedPipelineForhemanth, AdvancedPreProcessForHemanth,prepare_datasetsforhemanth
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset,DatasetDict
from PIL import Image
from typing import Dict, Optional,Tuple,Union, List
from transformers import AutoTokenizer,AutoFeatureExtractor,PreTrainedTokenizer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_split_dataset(
    path: str,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load and split the dataset into train, test, and evaluation sets.
    """
    try:
        dataset = load_dataset(path)
        logger.info(f"Dataset loaded successfully from {path}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    if isinstance(dataset, DatasetDict):
        dataset = dataset['train']  # Assuming the dataset has a 'train' split

    # Split the dataset
    train_test_split = dataset.train_test_split(test_size=sum(split_ratios[1:]))
    test_eval_split = train_test_split['test'].train_test_split(
        test_size=split_ratios[2]/(split_ratios[1]+split_ratios[2])
    )

    return train_test_split['train'], test_eval_split['train'], test_eval_split['test']

def create_augmentation_pipeline() -> A.Compose:
    """
    Create an image augmentation pipeline.
    """
    return A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ])

def preprocess_image(
    example: Dict,
    image_column: str,
    label_column: str,
    augment: bool = False,
    tokenizer: Optional[AutoTokenizer] = None
) -> Dict:
    """
    Preprocess a single example from the dataset.
    """
    image = Image.open(example[image_column]).convert('RGB')
    image_np = np.array(image)

    if augment:
        augmentation = create_augmentation_pipeline()
        augmented = augmentation(image=image_np)
        image_np = augmented['image']

    if tokenizer:
        inputs = tokenizer(image_np, return_tensors="pt", padding="max_length", truncation=True)
        inputs['pixel_values'] = inputs.pop('input_ids')
    else:
        inputs = {'pixel_values': torch.tensor(image_np.transpose(2, 0, 1)).unsqueeze(0)}

    inputs['label'] = torch.tensor(example[label_column])
    return inputs

def prepare_dataset(
    dataset: Dataset,
    image_column: str,
    label_column: str,
    tokenizer: Optional[AutoTokenizer] = None,
    augment: bool = False
) -> Dataset:
    """
    Prepare the dataset by applying preprocessing to each example.
    """
    return dataset.map(
        lambda example: preprocess_image(example, image_column, label_column, augment, tokenizer),
        remove_columns=dataset.column_names
    )

def create_data_loaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    eval_dataset: Dataset,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoader objects for train, test, and evaluation datasets.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

    return train_loader, test_loader, eval_loader

def pre_processing_image(
    dataset_path: str,
    image_column: str,
    label_column: str,
    batch_size: int = 32
):
    """
    Main function to orchestrate the data loading and preprocessing pipeline.
    """
    # Load and split the dataset
    train_dataset, test_dataset, eval_dataset = load_and_split_dataset(
        dataset_path, image_column, label_column
    )

    # Prepare datasets
    train_dataset = prepare_dataset(train_dataset, image_column, label_column, augment=True)
    test_dataset = prepare_dataset(test_dataset, image_column, label_column)
    eval_dataset = prepare_dataset(eval_dataset, image_column, label_column)

    # Create data loaders
    train_loader, test_loader, eval_loader = create_data_loaders(
        train_dataset, test_dataset, eval_dataset, batch_size
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")

    return train_loader, test_loader, eval_loader

class ImageProcessor:
    def __init__(
        self,
        model_type: str = "image",
        pretrained_model_name_or_path: str = "google/vit-base-patch16-224",
        cache_dir: str = r"C:\Users\heman\Desktop\Coding\data"
    ) -> None:
        self.image_preprocessing = AdvancedPreProcessForHemanth(
            model_type=model_type,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir
        )
        self.image_preprocessing = self.image_preprocessing.process_data()
        self.augmentations = self._get_augmentations()

    def _get_augmentations(self) -> A.Compose:
        return A.Compose([
            A.OneOf([
                A.AdvancedBlur(p=0.5),
                A.Blur(p=0.5),
                A.GaussianBlur(p=0.5),
                A.MedianBlur(p=0.5),
                A.MotionBlur(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.CLAHE(p=0.5),
                A.Equalize(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.ChannelDropout(p=0.5),
                A.ChannelShuffle(p=0.5),
                A.ColorJitter(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.RGBShift(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.ISONoise(p=0.5),
                A.MultiplicativeNoise(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.Downscale(p=0.5),
                A.Emboss(p=0.5),
                A.Sharpen(p=0.5),
                A.UnsharpMask(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.RandomFog(p=0.5),
                A.RandomRain(p=0.5),
                A.RandomShadow(p=0.5),
                A.RandomSnow(p=0.5),
                A.RandomSunFlare(p=0.5),
            ], p=0.5),
        ])

    def load_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def apply_augmentations(self, image: Image.Image) -> Image.Image:
        augmented = self.augmentations(image=np.array(image))
        return Image.fromarray(augmented['image'])

    def preprocess_image(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        return self.image_preprocessing(image)

    def process_image(self, image_path: str) -> Dict[str, torch.Tensor]:
        image = self.load_image(image_path)
        augmented_image = self.apply_augmentations(image)
        return self.preprocess_image(augmented_image)






class ImageDatasetProcessor:
    def __init__(
        self, 
        dataset_path: str, 
        batch_size: int = 32,
        model_type: str = "image",
        target_column: str='image',
        pretrained_model_name_or_path: str = "google/vit-base-patch16-224",
        cache_dir: str = r"C:\Users\heman\Desktop\Coding\data",
    ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.image_preprocessing = AdvancedPreProcessForHemanth(
            model_type=model_type,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir
        ).process_data()
        self.augmentations = self._get_augmentations()
        self.cache_dir=cache_dir
        self.target_column=target_column
    def _get_augmentations(self) -> A.Compose:
        return A.Compose([
            A.OneOf([
                A.AdvancedBlur(p=0.5),
                A.Blur(p=0.5),
                A.GaussianBlur(p=0.5),
                A.MedianBlur(p=0.5),
                A.MotionBlur(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.CLAHE(p=0.5),
                A.Equalize(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.ChannelDropout(p=0.5),
                A.ChannelShuffle(p=0.5),
                A.ColorJitter(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.RGBShift(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.ISONoise(p=0.5),
                A.MultiplicativeNoise(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.Downscale(p=0.5),
                A.Emboss(p=0.5),
                A.Sharpen(p=0.5),
                A.UnsharpMask(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.RandomFog(p=0.5),
                A.RandomRain(p=0.5),
                A.RandomShadow(p=0.5),
                A.RandomSnow(p=0.5),
                A.RandomSunFlare(p=0.5),
            ], p=0.5),
        ])

    def load_and_process_dataset(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        dataset = self._load_dataset()
        train_dataset, test_dataset, eval_dataset = self._split_dataset(dataset)
        train_dataset = self._process_dataset(train_dataset)
        test_dataset = self._process_dataset(test_dataset)
        eval_dataset = self._process_dataset(eval_dataset)
        train_loader = self._create_dataloader(train_dataset)
        test_loader = self._create_dataloader(test_dataset)
        eval_loader = self._create_dataloader(eval_dataset)
        return train_loader, test_loader, eval_loader

    def _load_dataset(self) -> Dataset:
        logger.info(f"Loading dataset from {self.dataset_path}")
        try:
            dataset = load_dataset(self.dataset_path, cache_dir=self.cache_dir)
            return dataset[list(dataset.keys())[0]]
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def _split_dataset(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        logger.info("Splitting dataset into train, test, and eval sets")
        train_test = dataset.train_test_split(test_size=0.2)
        test_eval = train_test['test'].train_test_split(test_size=0.5)
        return train_test['train'], test_eval['train'], test_eval['test']

    def _process_dataset(self, dataset: Dataset) -> Dataset:
        logger.info("Processing dataset")
        return dataset.map(self.process_image, remove_columns=dataset.column_names)

    def process_image(self, example: Dict) -> Dict[str, torch.Tensor]:
        image = example[self.target_column]
        if isinstance(image, str):
            image = self.load_image(image)
        augmented_image = self.apply_augmentations(image)
        processed_image = self.preprocess_image(augmented_image)
        return {
            "pixel_values": processed_image['pixel_values'],
            "label": torch.tensor(example['label'], dtype=torch.long)
        }

    def load_image(self, image_path: str) -> Image.Image:
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return Image.new('RGB', (224, 224))

    def apply_augmentations(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, Image.Image):
            image = np.array(image)
        augmented = self.augmentations(image=image)
        return Image.fromarray(augmented['image'])

    def preprocess_image(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        return self.image_preprocessing(image)

    def _create_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    

def load_and_split_dataset_test(
    path: str,
    train_size: float = 0.8,
    test_size: float = 0.1,
    eval_size: float = 0.1,
    seed: int = 42
) -> Dict[str, Dataset]:
    """
    Load and split the dataset into train, test, and eval sets.
    """
    try:
        dataset = load_dataset(path)
        logger.info(f"Dataset loaded successfully from {path}")
    except Exception as e:
        logger.error(f"Failed to load dataset from {path}: {str(e)}")
        raise

    if isinstance(dataset, Dataset):
        dataset = dataset.train_test_split(
            test_size=test_size + eval_size,
            shuffle=True,
            seed=seed
        )
        test_eval = dataset['test'].train_test_split(
            test_size=eval_size / (test_size + eval_size),
            shuffle=True,
            seed=seed
        )
        return {
            'train': dataset['train'],
            'test': test_eval['train'],
            'eval': test_eval['test']
        }
    elif 'train' in dataset:
        train_test = dataset['train'].train_test_split(
            test_size=test_size + eval_size,
            shuffle=True,
            seed=seed
        )
        test_eval = train_test['test'].train_test_split(
            test_size=eval_size / (test_size + eval_size),
            shuffle=True,
            seed=seed
        )
        return {
            'train': train_test['train'],
            'test': test_eval['train'],
            'eval': test_eval['test']
        }
    else:
        logger.error("Unsupported dataset format")
        raise ValueError("Unsupported dataset format")

def get_augmentations() -> A.Compose:
    """
    Define image augmentations.
    """
    return A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.GaussNoise(),
            A.ISONoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.PiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        ToTensorV2(),
    ])

def preprocess_function(
    examples: Dict[str, Union[str, int]],
    tokenizer: AutoTokenizer,
    feature_extractor: AutoFeatureExtractor,
    image_column: str,
    label_column: str,
    augmentations: Optional[A.Compose] = None
) -> Dict[str, Union[List, np.ndarray]]:
    """
    Preprocess the dataset examples.
    """
    images = [img for img in examples[image_column]]
    labels = examples[label_column]

    if augmentations:
        augmented = [augmentations(image=img)['image'] for img in images]
        pixel_values = feature_extractor(images=augmented, return_tensors="pt")['pixel_values']
    else:
        pixel_values = feature_extractor(images=images, return_tensors="pt")['pixel_values']

    encoded_inputs = tokenizer(labels, padding="max_length", truncation=True, return_tensors="pt")

    return {
        "pixel_values": pixel_values,
        "input_ids": encoded_inputs['input_ids'],
        "attention_mask": encoded_inputs['attention_mask'],
        "labels": labels
    }


def create_dataloaders_image(
    datasets: Dict[str, Dataset],
    tokenizer: AutoTokenizer,
    feature_extractor: AutoFeatureExtractor,
    image_column: str,
    label_column: str,
    batch_size: int = 32
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train, test, and eval datasets.
    """
    augmentations = get_augmentations()

    processed_datasets = {}
    for split, dataset in datasets.items():
        processed_datasets[split] = dataset.map(
            lambda examples: preprocess_function(
                examples,
                tokenizer,
                feature_extractor,
                image_column,
                label_column,
                augmentations if split == 'train' else None
            ),
            batched=True,
            remove_columns=dataset.column_names
        )
        processed_datasets[split].set_format(type="torch")

    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train')
        )
        for split, dataset in processed_datasets.items()
    }

    return dataloaders

def pre_processing_image_text(
    tokenizer: PreTrainedTokenizer,
    feature_extractor: AutoFeatureExtractor,
    dataset_path: str,
    image_column: str,
    label_column: str,
    batch_size: int = 32,
   
):
    """
    Main function to load, preprocess, and create dataloaders.
    """
    datasets = load_and_split_dataset_test(dataset_path, image_column, label_column)

    dataloaders = create_dataloaders_image(
        datasets,
        tokenizer,
        feature_extractor,
        image_column,
        label_column,
        batch_size
    )

    logger.info("DataLoaders created successfully")
    return dataloaders
