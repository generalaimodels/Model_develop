import sys
from pathlib import Path
file=Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
from typing import Union, Tuple, List, Optional, Dict, Any
import numpy as np
import torch
import librosa
import torchaudio
import soundfile as sf
import logging

from FAST_ANALYSIS import AdvancedPipelineForhemanth, AdvancedPreProcessForHemanth
from datasets import load_dataset, Dataset
from transformers import AutoFeatureExtractor,PreTrainedTokenizer
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioDatasetProcessor:
    def __init__(
        self,
        dataset_path: str,
        feature_extractor: AutoFeatureExtractor,
        audio_column: str,
        tokenizer: PreTrainedTokenizer=None,
        label_column: Optional[str] = None,
        transcript_column: Optional[str] = None,
        target_sample_rate: int = 16000,
        max_duration: float = 10.0,
    ):
        self.dataset_path = dataset_path
        self.audio_column = audio_column
        self.label_column = label_column
        self.transcript_column = transcript_column
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        self.feature_extractor = feature_extractor
        self.tokenizer=tokenizer
    def load_dataset(self) -> Dataset:
        try:
            dataset = load_dataset(self.dataset_path)
            logger.info(f"Dataset loaded successfully from {self.dataset_path}")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def preprocess_audio(self, example: Dict[str, Any]) -> Dict[str, Any]:
        audio = example[self.audio_column]
        
        # Ensure mono channel
        if len(audio["array"].shape) > 1:
            audio["array"] = np.mean(audio["array"], axis=1)
        
        # Resample if necessary
        if audio["sampling_rate"] != self.target_sample_rate:
            audio["array"] = librosa.resample(
                audio["array"],
                orig_sr=audio["sampling_rate"],
                target_sr=self.target_sample_rate
            )
            audio["sampling_rate"] = self.target_sample_rate

        # Apply fixed length (pad or truncate)
        target_length = int(self.max_duration * self.target_sample_rate)
        if len(audio["array"]) > target_length:
            audio["array"] = audio["array"][:target_length]
        elif len(audio["array"]) < target_length:
            padding = np.zeros(target_length - len(audio["array"]))
            audio["array"] = np.concatenate([audio["array"], padding])

        inputs = self.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        )

        example["input_audio"] = inputs.input_values[0]

        if self.label_column:
            example["label"] = example[self.label_column]

        if self.transcript_column:
            # Assuming you have a tokenizer defined
            tokenizer = self.tokenizer
            tokens = tokenizer(example[self.transcript_column], return_tensors="pt")
            example["input_ids"] = tokens.input_ids[0]
            example["labels"] = tokens.input_ids[0]
            
            # Create masked_ids (you may want to customize this based on your needs)
            masked_ids = tokens.input_ids[0].clone()
            mask_prob = 0.15
            mask = torch.rand(masked_ids.shape) < mask_prob
            masked_ids[mask] = tokenizer.mask_token_id
            example["masked_ids"] = masked_ids

        return example

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        try:
            processed_dataset = dataset.map(
                self.preprocess_audio,
                remove_columns=dataset.column_names,
                num_proc=4
            )
            logger.info("Dataset preprocessing completed successfully")
            return processed_dataset
        except Exception as e:
            logger.error(f"Error preprocessing dataset: {str(e)}")
            raise

    def split_dataset(self, dataset: Dataset) -> Dict[str, Dataset]:
        try:
            train_testval = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
            test_eval = train_testval["test"].train_test_split(test_size=0.5, shuffle=True, seed=42)
            
            split_dataset = {
                "train": train_testval["train"],
                "test": test_eval["train"],
                "eval": test_eval["test"]
            }
            
            logger.info("Dataset split completed: 80% train, 10% test, Remaining eval")
            return split_dataset
        except Exception as e:
            logger.error(f"Error splitting dataset: {str(e)}")
            raise

    def create_dataloaders(
        self,
        split_dataset: Dict[str, Dataset],
        batch_size: int,
        num_workers: int
    ) -> Dict[str, DataLoader]:
        try:
            dataloaders = {
                split: DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=(split == "train"),
                    num_workers=num_workers,
                    pin_memory=True
                )
                for split, dataset in split_dataset.items()
            }
            logger.info("DataLoaders created successfully")
            return dataloaders
        except Exception as e:
            logger.error(f"Error creating DataLoaders: {str(e)}")
            raise

class AudioProcessor:
    def __init__(
        self,
        model_type: str = "audio",
        pretrained_model_name_or_path: str = "superb/wav2vec2-base-superb-sid",
        cache_dir: str = r"C:\Users\heman\Desktop\Coding\data",
        target_sample_rate: int = 16000
    ) -> None:
        self.target_sample_rate = target_sample_rate
        self.audio_preprocessing = AdvancedPreProcessForHemanth(
            model_type=model_type,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir
        )
        self.audio_preprocessing = self.audio_preprocessing.process_data()

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        loaders = [
            (librosa.load, {"sr": self.target_sample_rate, "mono": True}),
            (torchaudio.load, {}),
            (sf.read, {})
        ]

        for loader, kwargs in loaders:
            try:
                audio, sr = loader(audio_path, **kwargs)
                if isinstance(audio, torch.Tensor):
                    audio = audio.numpy()
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=0 if audio.shape[0] > audio.shape[1] else 1)
                if sr != self.target_sample_rate:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sample_rate)
                return audio, self.target_sample_rate
            except Exception:
                continue

        raise RuntimeError(f"Failed to load audio file: {audio_path}")

    def extract_mfcc_features(
        self,
        audio_data: np.ndarray,
        num_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        mfccs = librosa.feature.mfcc(
            y=audio_data, sr=self.target_sample_rate, n_mfcc=num_mfcc,
            n_fft=n_fft, hop_length=hop_length
        )
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        energy = librosa.feature.rms(y=audio_data, frame_length=n_fft, hop_length=hop_length)
        features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs, energy), axis=0)
        return features
    
    def standardize_features(self, features: np.ndarray,target_length: int = 40) -> np.ndarray:
        if features.shape[1] > target_length:
            return features[:, :target_length]
        else:
            padding = target_length - features.shape[1]
            return np.pad(features, ((0, 0), (0, padding)), mode='constant')


    def apply_augmentations(self, audio: np.ndarray) -> np.ndarray:
        # Implement audio augmentations here
        # For example, you could add noise, shift pitch, change speed, etc.
        return audio  # Placeholder, return original audio for now

    def preprocess_audio(self, audio: np.ndarray) -> Dict[str, torch.Tensor]:
        tensor_input = torch.from_numpy(audio).float()
        return self.audio_preprocessing(tensor_input, sampling_rate=self.target_sample_rate)

    def process_audio(self, audio_path: str) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        audio, _ = self.load_audio(audio_path)
        augmented_audio = self.apply_augmentations(audio)
        mfcc_features = self.extract_mfcc_features(augmented_audio)
        advanced_features = self.preprocess_audio(augmented_audio)
        
        return {
            'mfcc_features': self.standardize_features(mfcc_features),
            'advanced_features': advanced_features,
            'raw_audio': augmented_audio
        }

# def main(audio_paths: List[str]) -> List[Dict[str, Union[np.ndarray, torch.Tensor]]]:
#     processor = AudioProcessor()
#     results = []
#     for audio_path in audio_paths:
#         try:
#             result = processor.process_audio(audio_path)
#             results.append(result)
#         except Exception as e:
#             print(f"Error processing {audio_path}: {str(e)}")
#     return results

# if __name__ == "__main__":
#     audio_paths = [
#         r"C:\Users\heman\Desktop\Coding\output1\audio\input_1.flac",
#         # r"C:\Users\heman\Desktop\Coding\output1\audio\input_2.flac",
#         # r"C:\Users\heman\Desktop\Coding\output1\audio\input_3.flac",
#     ]
#     processed_audios = main(audio_paths)
#     for i, output in enumerate(processed_audios):
#         print(f"Processed audio {i + 1}:")
#         print(f"  MFCC Features shape: {output['mfcc_features'].shape}")
#         print(f"  Advanced Features shape: {output['advanced_features']['input_values'][0].shape}")
#         print(f"  Raw Audio shape: {output['raw_audio'].shape}")