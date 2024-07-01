import gc
import threading
import psutil
import torch
import warnings
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Union, List, Type
from pathlib import Path
from transformers.models.auto.auto_factory import _BaseAutoBackboneClass
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          MODEL_FOR_MASK_GENERATION_MAPPING,
                          MODEL_FOR_KEYPOINT_DETECTION_MAPPING,
                          MODEL_FOR_TEXT_ENCODING_MAPPING,
                          MODEL_FOR_IMAGE_TO_IMAGE_MAPPING,
                          MODEL_MAPPING,
                          MODEL_FOR_PRETRAINING_MAPPING,
                          MODEL_WITH_LM_HEAD_MAPPING,
                          MODEL_FOR_CAUSAL_LM_MAPPING,
                          MODEL_FOR_MASKED_LM_MAPPING,
                          MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
                          MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
                          MODEL_FOR_QUESTION_ANSWERING_MAPPING,
                          MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
                          MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING,
                          MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
                          MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
                          MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
                          MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
                          MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
                          MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,
                          MODEL_FOR_IMAGE_SEGMENTATION_MAPPING,
                          MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
                          MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING,
                          MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING,
                          MODEL_FOR_OBJECT_DETECTION_MAPPING,
                          MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING,
                          MODEL_FOR_DEPTH_ESTIMATION_MAPPING,
                          MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING,
                          MODEL_FOR_VISION_2_SEQ_MAPPING,
                          MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
                          MODEL_FOR_CTC_MAPPING,
                           MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
                            MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING,
                             MODEL_FOR_AUDIO_XVECTOR_MAPPING,
                             MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING,
                             MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING,
                             MODEL_FOR_BACKBONE_MAPPING,
                             MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
                          )
from typing import Dict, Optional, Union, Tuple,List
import plotly.subplots as sp

from typing import Any, Union, List, Type
from pathlib import Path

class ModelLoadingError(Exception):
    """Custom exception for model loading errors."""
    pass

class _BaseAutoModelClass:
    _model_mapping: dict

    @classmethod
    def from_config(cls, config: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], *model_args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

def auto_class_update(cls: Type[_BaseAutoModelClass], head_doc: str = "", checkpoint_for_example: str = "") -> Type[_BaseAutoModelClass]:
    cls.__doc__ = f"{head_doc}\n\nCheckpoint for example: {checkpoint_for_example}"
    return cls

class AutoModelForMaskGeneration(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASK_GENERATION_MAPPING

class AutoModelForKeypointDetection(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_KEYPOINT_DETECTION_MAPPING

class AutoModelForTextEncoding(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_ENCODING_MAPPING

class AutoModelForImageToImage(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_TO_IMAGE_MAPPING

class AutoModel(_BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING

AutoModel = auto_class_update(AutoModel)

class AutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PRETRAINING_MAPPING

AutoModelForPreTraining = auto_class_update(AutoModelForPreTraining, head_doc="pretraining")

class _AutoModelWithLMHead(_BaseAutoModelClass):
    _model_mapping = MODEL_WITH_LM_HEAD_MAPPING

_AutoModelWithLMHead = auto_class_update(_AutoModelWithLMHead, head_doc="language modeling")

class AutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING

AutoModelForCausalLM = auto_class_update(AutoModelForCausalLM, head_doc="causal language modeling")

class AutoModelForMaskedLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASKED_LM_MAPPING

AutoModelForMaskedLM = auto_class_update(AutoModelForMaskedLM, head_doc="masked language modeling")

class AutoModelForSeq2SeqLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING

AutoModelForSeq2SeqLM = auto_class_update(
    AutoModelForSeq2SeqLM,
    head_doc="sequence-to-sequence language modeling",
    checkpoint_for_example="google/t5-base"
)

class AutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

AutoModelForSequenceClassification = auto_class_update(
    AutoModelForSequenceClassification, head_doc="sequence classification"
)

class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING

AutoModelForQuestionAnswering = auto_class_update(AutoModelForQuestionAnswering, head_doc="question answering")

class AutoModelForTableQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING

AutoModelForTableQuestionAnswering = auto_class_update(
    AutoModelForTableQuestionAnswering,
    head_doc="table question answering",
    checkpoint_for_example="google/tapas-base-finetuned-wtq"
)

class AutoModelForVisualQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING

AutoModelForVisualQuestionAnswering = auto_class_update(
    AutoModelForVisualQuestionAnswering,
    head_doc="visual question answering",
    checkpoint_for_example="dandelin/vilt-b32-finetuned-vqa"
)

class AutoModelForDocumentQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING

AutoModelForDocumentQuestionAnswering = auto_class_update(
    AutoModelForDocumentQuestionAnswering,
    head_doc="document question answering",
    checkpoint_for_example="impira/layoutlm-document-qa", 
)

class AutoModelForTokenClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING

AutoModelForTokenClassification = auto_class_update(AutoModelForTokenClassification, head_doc="token classification")

class AutoModelForMultipleChoice(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MULTIPLE_CHOICE_MAPPING

AutoModelForMultipleChoice = auto_class_update(AutoModelForMultipleChoice, head_doc="multiple choice")

class AutoModelForNextSentencePrediction(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING

AutoModelForNextSentencePrediction = auto_class_update(
    AutoModelForNextSentencePrediction, head_doc="next sentence prediction"
)

class AutoModelForImageClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING

AutoModelForImageClassification = auto_class_update(AutoModelForImageClassification, head_doc="image classification")

class AutoModelForZeroShotImageClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING

AutoModelForZeroShotImageClassification = auto_class_update(
    AutoModelForZeroShotImageClassification, head_doc="zero-shot image classification"
)

class AutoModelForImageSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_SEGMENTATION_MAPPING

AutoModelForImageSegmentation = auto_class_update(AutoModelForImageSegmentation, head_doc="image segmentation")

class AutoModelForSemanticSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING

AutoModelForSemanticSegmentation = auto_class_update(
    AutoModelForSemanticSegmentation, head_doc="semantic segmentation"
)

class AutoModelForUniversalSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING

AutoModelForUniversalSegmentation = auto_class_update(
    AutoModelForUniversalSegmentation, head_doc="universal image segmentation"
)

class AutoModelForInstanceSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING

AutoModelForInstanceSegmentation = auto_class_update(
    AutoModelForInstanceSegmentation, head_doc="instance segmentation"
)

class AutoModelForObjectDetection(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_OBJECT_DETECTION_MAPPING

AutoModelForObjectDetection = auto_class_update(AutoModelForObjectDetection, head_doc="object detection")

class AutoModelForZeroShotObjectDetection(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING

AutoModelForZeroShotObjectDetection = auto_class_update(
    AutoModelForZeroShotObjectDetection, head_doc="zero-shot object detection"
)

class AutoModelForDepthEstimation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_DEPTH_ESTIMATION_MAPPING

AutoModelForDepthEstimation = auto_class_update(AutoModelForDepthEstimation, head_doc="depth estimation")

class AutoModelForVideoClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING

AutoModelForVideoClassification = auto_class_update(AutoModelForVideoClassification, head_doc="video classification")

class AutoModelForVision2Seq(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_VISION_2_SEQ_MAPPING

AutoModelForVision2Seq = auto_class_update(AutoModelForVision2Seq, head_doc="vision-to-text modeling")

class AutoModelForAudioClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING

AutoModelForAudioClassification = auto_class_update(AutoModelForAudioClassification, head_doc="audio classification")

class AutoModelForCTC(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CTC_MAPPING

AutoModelForCTC = auto_class_update(AutoModelForCTC, head_doc="connectionist temporal classification")

class AutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING

AutoModelForSpeechSeq2Seq = auto_class_update(
    AutoModelForSpeechSeq2Seq, head_doc="sequence-to-sequence speech-to-text modeling"
)

class AutoModelForAudioFrameClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING

AutoModelForAudioFrameClassification = auto_class_update(
    AutoModelForAudioFrameClassification, head_doc="audio frame (token) classification"
)

class AutoModelForAudioXVector(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_XVECTOR_MAPPING

AutoModelForAudioXVector = auto_class_update(AutoModelForAudioXVector, head_doc="audio retrieval via x-vector")

class AutoModelForTextToSpectrogram(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING

AutoModelForTextToSpectrogram = auto_class_update(AutoModelForTextToSpectrogram, head_doc="text to spectrogram modeling")

class AutoModelForTextToWaveform(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING

AutoModelForTextToWaveform = auto_class_update(AutoModelForTextToWaveform, head_doc="text to waveform modeling")

class AutoBackbone(_BaseAutoBackboneClass):
    _model_mapping = MODEL_FOR_BACKBONE_MAPPING

AutoBackbone = auto_class_update(AutoBackbone, head_doc="backbone modeling")

class AutoModelForMaskedImageModeling(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING

AutoModelForMaskedImageModeling = auto_class_update(AutoModelForMaskedImageModeling, head_doc="masked image modeling")

class AutoModelWithLMHead(_AutoModelWithLMHead):
    @classmethod
    def from_config(cls, config: Any) -> Any:
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], *model_args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)



class ModelLoadingError_test(Exception):
    """Custom exception for model loading errors."""
    pass

class AutoModelLoader_test:
    model_classes = {
        "causal_lm": AutoModelForCausalLM,
        "masked_lm": AutoModelForMaskedLM,
        "seq2seq_lm": AutoModelForSeq2SeqLM,
        "sequence_classification": AutoModelForSequenceClassification,
        "question_answering": AutoModelForQuestionAnswering,
        "table_question_answering": AutoModelForTableQuestionAnswering,
        "visual_question_answering": AutoModelForVisualQuestionAnswering,
        "document_question_answering": AutoModelForDocumentQuestionAnswering,
        "token_classification": AutoModelForTokenClassification,
        "multiple_choice": AutoModelForMultipleChoice,
        "next_sentence_prediction": AutoModelForNextSentencePrediction,
        "image_classification": AutoModelForImageClassification,
        "zero_shot_image_classification": AutoModelForZeroShotImageClassification,
        "image_segmentation": AutoModelForImageSegmentation,
        "semantic_segmentation": AutoModelForSemanticSegmentation,
        "universal_segmentation": AutoModelForUniversalSegmentation,
        "instance_segmentation": AutoModelForInstanceSegmentation,
        "object_detection": AutoModelForObjectDetection,
        "zero_shot_object_detection": AutoModelForZeroShotObjectDetection,
        "depth_estimation": AutoModelForDepthEstimation,
        "video_classification": AutoModelForVideoClassification,
        "vision2seq": AutoModelForVision2Seq,
        "audio_classification": AutoModelForAudioClassification,
        "ctc": AutoModelForCTC,
        "speech_seq2seq": AutoModelForSpeechSeq2Seq,
        "audio_frame_classification": AutoModelForAudioFrameClassification,
        "audio_xvector": AutoModelForAudioXVector,
        "text_to_spectrogram": AutoModelForTextToSpectrogram,
        "text_to_waveform": AutoModelForTextToWaveform,
        "backbone": AutoBackbone,
        "masked_image_modeling": AutoModelForMaskedImageModeling,
        "mask_generation": AutoModelForMaskGeneration,
        "keypoint_detection": AutoModelForKeypointDetection,
        "text_encoding": AutoModelForTextEncoding,
        "image_to_image": AutoModelForImageToImage,
        "pretraining": AutoModelForPreTraining,
        "lm_head": AutoModelWithLMHead,
    }

    @classmethod
    def load_model(cls, model_type: str, model_name_or_path: Union[str, Path], *model_args: Any, **kwargs: Any) -> _BaseAutoModelClass:
        """
        Function to load a transformers model.

        Args:
            model_type (str): The type of the model (e.g., 'causal_lm', 'masked_lm').
            model_name_or_path (Union[str, Path]): The name or path of the model.

        Returns:
            model (_BaseAutoModelClass): The loaded model.
        """
        if model_type not in cls.model_classes:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = cls.model_classes[model_type]
        print(model_class)

        try:
            model = model_class.from_pretrained(model_name_or_path, *model_args, **kwargs)
        except Exception as e:
            raise ModelLoadingError(f"Error loading model: {e}")

        return model



def get_device() -> torch.device:
    """
    Automatically set the device to CPU, CUDA, or other available hardware backends.
    
    Returns:
        torch.device: The chosen device.
    """
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("\n" + "="*30)
        print("| Using CUDA (GPU support) |")
        print("="*30 + "\n")
        return torch.device('cuda')
    
    # Check for other backends like MCOs or TPUs
    # Note: PyTorch doesn't have native TPU support without third-party libraries like PyTorch/XLA
    # We'll check for TPU availability using the xla library as an example
    try:
        import torch_xla.core.xla_model as xm
        print("\n" + "="*30)
        print("| Using TPU |")
        print("="*30 + "\n")
        return xm.xla_device()
    except ImportError:
        pass  # TPU not available or xla not installed
    
    # Fallback to CPU
    print("\n" + "="*30)
    print("| Using CPU |")
    print("="*30 + "\n")
    return torch.device('cpu')

def calculate_model_parameters(model: AutoModelForCausalLM) -> None:
    """
    Calculate and print the model parameters.

    Args:
        model (AutoModelForCausalLM): The loaded model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"╔══════════════════════════════════╗")
    print(f"║        Total Parameters          ║")
    print(f"╠══════════════════════════════════╣")
    print(f"║ Total parameter: {total_params:,}   ║")
    print(f"║ Trainable parameters:{trainable_params:,} ║")
    print(f"║ Non-trainable parameters: {total_params - trainable_params:,}      ║")
    print(f"╚══════════════════════════════════╝")
    print()

    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║                            Calculations                                  ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    for name, param in model.named_parameters():
        param_count = param.numel()
        trainable = param.requires_grad
        print(f"║ {name:<60} │ {param_count:>12,} parameters │ Trainable: {str(trainable):<5}║")

    print(f"╠══════════════════════════════════════════════════════════════════════════╣")
    print(f"║ Total parameters calculation:                                            ║")
    print(f"║ sum(p.numel() for p in model.parameters()) = {total_params:,}               ║")
    print(f"║ Trainable parameters calculation:                                        ║")
    print(f"║ sum(p.numel() for p in model.parameters() if p.requires_grad) = {trainable_params:,}║")
    print(f"╚══════════════════════════════════════════════════════════════════════════╝")
def b2mb(bytes: int) -> float:
    """
    Convert bytes to megabytes.

    Args:
        bytes (int): The number of bytes.

    Returns:
        float: The equivalent value in megabytes.
    """
    return bytes / 1024 / 1024

class ResourceMonitor:
    def __init__(self):
        self.begin = 0
        self.end = 0
        self.peak = 0
        self.used = 0
        self.peaked = 0
        self.cpu_begin = 0
        self.cpu_end = 0
        self.cpu_peak = -1
        self.cpu_used = 0
        self.cpu_peaked = 0
        self.peak_monitoring = False
        self.process = psutil.Process()

    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        self.begin = torch.cuda.memory_allocated()
        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self) -> int:
        """
        Get resident set size memory for the current process.

        Returns:
            int: The memory used by the current process in bytes.
        """
        return self.process.memory_info().rss

    def peak_monitor_func(self) -> None:
        """
        Monitor the peak CPU memory usage.
        """
        while self.peak_monitoring:
            current_cpu_usage = self.cpu_mem_used()
            self.cpu_peak = max(current_cpu_usage, self.cpu_peak)

    def __exit__(self, *exc) -> None:
        self.peak_monitoring = False
        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)
        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        print("╔═════════════════════════════════════════════════╗")
        print("║               Memory Usage                      ║")
        print("╠═════════════════════════════════════════════════╣")
        print(f"║ Memory used:  {self.used:>10} MB                ║")
        print(f"║ Memory peaked: {self.peaked:>9} MB               ║")
        print("╚═════════════════════════════════════════════════╝")
        
        print("╔═════════════════════════════════════════════════╗")
        print("║                 CPU Usage                       ║")
        print("╠═════════════════════════════════════════════════╣")
        print(f"║ CPU used:  {self.cpu_used:>13} MB                     ║")
        print(f"║ CPU peaked: {self.cpu_peaked:>12} MB                     ║")
        print("╚═════════════════════════════════════════════════╝")

def load_model_test(model_name_or_path: Union[str,List],*model_args, **kwargs) -> AutoModelForCausalLM:
    """
    Function to load a transformers model.
    
    Args:
      model_name_or_path (Union[str, Path]): The name or path of the model.

    Returns:
        model (AutoModelForCausalLM): The loaded model.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                *model_args, **kwargs)
    return model

def create_tokenizer(
    tokenizer_name_or_path: Union[str,List] ) -> AutoTokenizer:
    """
    Initializes and returns a tokenizer based on the specified pretrained model or path.

    Args:
        tokenizer_name_or_path (str): The name or path of the tokenizer's pretrained model.

    Returns:
        AutoTokenizer: The initialized tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    
    # Set special tokens if they are not already set
    special_tokens = {
        'pad_token': tokenizer.eos_token,
        'bos_token': tokenizer.eos_token,
        'eos_token': tokenizer.eos_token,
        'unk_token': tokenizer.eos_token,
        'sep_token': tokenizer.eos_token,
        'cls_token': tokenizer.eos_token,
        'mask_token':tokenizer.eos_token
    }
    for token_name, token_value in special_tokens.items():
        if getattr(tokenizer, f"{token_name}_id") is None:
            setattr(tokenizer, token_name, token_value)
    
    return tokenizer

def load_model(
    model_name: str,
    max_memory: Dict[Union[int, str], str],
    quantize: Optional[bool] = True,
    device_map: str = "auto"
    ) -> AutoModelForCausalLM:
    """
    Load the model with optional quantization.

    Args:
        model_name (str): The name of the model to load.
        max_memory (Dict[Union[int, str], str]): Max memory configuration.
        quantize (Optional[bool], optional): Whether to quantize the model. Defaults to False.
        device_map (str, optional): The device map configuration. Defaults to "auto".

    Returns:
        AutoModelForCausalLM: The loaded model.
    """
    device = get_device()
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            max_memory=max_memory
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            device_map=device_map,
            max_memory=max_memory
        ).to(device)
    return model

def load_model_and_tokenizer(
    model_name: str,
    device_map: str,
    max_memory: Dict[Union[int, str], str]
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the model and tokenizer.

    Args:
        model_name (str): The name of the model to load.
        device_map (str): The device map configuration.
        max_memory (Dict[Union[int, str], str]): Max memory configuration.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.
    """
    with ResourceMonitor():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            max_memory=max_memory
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            device_map=device_map,
            max_memory=max_memory
        )

    return model, tokenizer

def get_model_io_dimensions(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text:str,device: str) -> Tuple[int, int]:
    """
    Determine the input and output dimensions of the model.

    Args:
        model (AutoModelForCausalLM): The loaded model.
        tokenizer (AutoTokenizer): The associated tokenizer.
        device (str): The device to perform computation on.

    Returns:
        Tuple[int, int]: The input and output dimensions of the model.
    """
    
    encoded_input = tokenizer.encode_plus(text, return_tensors="pt",truncation=True,max_length=512)
    input_dim = encoded_input['input_ids'].shape[-1] 
    print("Token's shape :",encoded_input['input_ids'].shape)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    
    model = model.to(device)

    with torch.no_grad():
        output = model(**encoded_input)
    
    print("Output shape :",output.logits.shape)
   
    output_dim = output.logits.shape[-1] 
    
    return input_dim, output_dim

def visualize_text_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str,
    num_tokens: int,
    device: torch.device
    ) -> None:
    """
    Visualize the model's text generation process.

    Args:
        model (AutoModelForCausalLM): The loaded model.
        tokenizer (AutoTokenizer): The associated tokenizer.
        input_text (str): The input text to generate from.
        num_tokens (int): The number of tokens to generate.
        device (torch.device): The device to perform computation on.
    """
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + num_tokens,
            do_sample=True,
            top_k=100,
            top_p=0.1,
            num_return_sequences=1
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_tokens = tokenizer.tokenize(generated_text)
    num_rows = num_tokens // 5 + 1
    fig, axs = plt.subplots(num_rows, 5, figsize=(10, 4 * num_rows))
    axs = axs.flatten()
    for i in range(num_tokens):
        token_embedding = model.model.embed_tokens.weight[output[0, i]].detach().cpu().numpy()
        axs[i].imshow(token_embedding.reshape(1, -1), cmap="viridis", aspect="auto")
        axs[i].set_title(f"{i}:{generated_tokens[i]}")
        axs[i].axis("off")
    plt.tight_layout()
    plt.show()

def visualize_text_generation_test(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str,
    num_tokens: int,
    device: torch.device
) -> None:
    """
    Visualize the model's text generation process.

    Args:
        model (AutoModelForCausalLM): The loaded model.
        tokenizer (AutoTokenizer): The associated tokenizer.
        input_text (str): The input text to generate from.
        num_tokens (int): The number of tokens to generate.
        device (torch.device): The device to perform computation on.
    """

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)


    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + num_tokens,
            do_sample=True,
            top_k=100,
            top_p=0.1,
            num_return_sequences=1
        )


    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)


    generated_tokens = tokenizer.tokenize(generated_text)


    num_rows = num_tokens // 5 + 1
    fig = sp.make_subplots(rows=num_rows, cols=5, subplot_titles=[f"{i}:{token}" for i, token in enumerate(generated_tokens[:num_tokens])])


    for i in range(num_tokens):

        token_embedding = model.embed_tokens.weight[output[0, i]].detach().cpu().numpy()

        fig.add_heatmap(z=token_embedding.reshape(1, -1), colorscale="Viridis", showscale=False, row=(i // 5) + 1, col=(i % 5) + 1)

    fig.update_layout(height=400 * num_rows, width=1000, title_text="Token Embeddings")


    fig.show()

def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str, max_length: int, num_beams: int) -> str:
    

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text











