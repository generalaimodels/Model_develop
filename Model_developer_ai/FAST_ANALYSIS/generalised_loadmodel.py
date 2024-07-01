import logging
from typing import Dict, Optional, Union
from transformers import AutoConfig
from transformers.models.auto.modeling_auto import (
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForDocumentQuestionAnswering,
    AutoModelForImageClassification,
    AutoModelForImageSegmentation,
    AutoModelForMaskedLM,
    AutoModelForMaskGeneration,
    AutoModelForObjectDetection,
    AutoModelForQuestionAnswering,
    AutoModelForSemanticSegmentation,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoModelForTableQuestionAnswering,
    AutoModelForTextToSpectrogram,
    AutoModelForTextToWaveform,
    AutoModelForTokenClassification,
    AutoModelForVideoClassification,
    AutoModelForVision2Seq,
    AutoModelForVisualQuestionAnswering,
    AutoModelForZeroShotImageClassification,
    AutoModelForZeroShotObjectDetection,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedModelLoader:
    """A class for loading advanced AI models with generalized, robust, and scalable functionality."""

    MODEL_MAPPING = {
        "audio_classification": AutoModelForAudioClassification,
        "causal_lm": AutoModelForCausalLM,
        "ctc": AutoModelForCTC,
        "document_qa": AutoModelForDocumentQuestionAnswering,
        "image_classification": AutoModelForImageClassification,
        "image_segmentation": AutoModelForImageSegmentation,
        "masked_lm": AutoModelForMaskedLM,
        "mask_generation": AutoModelForMaskGeneration,
        "object_detection": AutoModelForObjectDetection,
        "question_answering": AutoModelForQuestionAnswering,
        "semantic_segmentation": AutoModelForSemanticSegmentation,
        "seq2seq_lm": AutoModelForSeq2SeqLM,
        "sequence_classification": AutoModelForSequenceClassification,
        "speech_seq2seq": AutoModelForSpeechSeq2Seq,
        "table_qa": AutoModelForTableQuestionAnswering,
        "text_to_spectrogram": AutoModelForTextToSpectrogram,
        "text_to_waveform": AutoModelForTextToWaveform,
        "token_classification": AutoModelForTokenClassification,
        "video_classification": AutoModelForVideoClassification,
        "vision2seq": AutoModelForVision2Seq,
        "visual_qa": AutoModelForVisualQuestionAnswering,
        "zero_shot_image_classification": AutoModelForZeroShotImageClassification,
        "zero_shot_object_detection": AutoModelForZeroShotObjectDetection,
    }

    @classmethod
    def load_model(cls, model_name: str, task: str, **kwargs) -> Optional[AutoModel]:
        """
        Load a model based on the given model name and task.

        Args:
            model_name (str): The name of the model to load.
            task (str): The task for which the model is intended.
            **kwargs: Additional keyword arguments to pass to the model loading function.

        Returns:
            Optional[AutoModel]: The loaded model, or None if loading fails
            
        example:
         "audio_classification": AutoModelForAudioClassification,
        "causal_lm": AutoModelForCausalLM,
        "ctc": AutoModelForCTC,
        "document_qa": AutoModelForDocumentQuestionAnswering,
        "image_classification": AutoModelForImageClassification,
        "image_segmentation": AutoModelForImageSegmentation,
        "masked_lm": AutoModelForMaskedLM,
        "mask_generation": AutoModelForMaskGeneration,
        "object_detection": AutoModelForObjectDetection,
        "question_answering": AutoModelForQuestionAnswering,
        "semantic_segmentation": AutoModelForSemanticSegmentation,
        "seq2seq_lm": AutoModelForSeq2SeqLM,
        "sequence_classification": AutoModelForSequenceClassification,
        "speech_seq2seq": AutoModelForSpeechSeq2Seq,
        "table_qa": AutoModelForTableQuestionAnswering,
        "text_to_spectrogram": AutoModelForTextToSpectrogram,
        "text_to_waveform": AutoModelForTextToWaveform,
        "token_classification": AutoModelForTokenClassification,
        "video_classification": AutoModelForVideoClassification,
        "vision2seq": AutoModelForVision2Seq,
        "visual_qa": AutoModelForVisualQuestionAnswering,
        "zero_shot_image_classification": AutoModelForZeroShotImageClassification,
        "zero_shot_object_detection": AutoModelForZeroShotObjectDetection,
        
        
        """
        try:
            logger.info(f"Loading model: {model_name} for task: {task}")
            config = AutoConfig.from_pretrained(model_name)
            
            if task in cls.MODEL_MAPPING:
                model_class = cls.MODEL_MAPPING[task]
                model = model_class.from_pretrained(model_name, config=config, **kwargs)
                logger.info(f"Successfully loaded model: {model_name}")
                return model
            else:
                logger.warning(f"Unknown task: {task}. Falling back to generic AutoModel.")
                return AutoModel.from_pretrained(model_name, config=config, **kwargs)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None

    @staticmethod
    def get_model_info(model: AutoModel) -> Dict[str, Union[str, int]]:
        """
        Get information about the loaded model.

        Args:
            model (AutoModel): The loaded model.

        Returns:
            Dict[str, Union[str, int]]: A dictionary containing model information.
        """
        return {
            "model_type": model.__class__.__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "device": next(model.parameters()).device.type,
        }

