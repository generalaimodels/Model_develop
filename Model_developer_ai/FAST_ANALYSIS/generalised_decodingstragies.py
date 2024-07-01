import logging
from typing import List, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_text_generalised(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: Union[str, List[str]],
    max_length: int = 20,
    min_length: int = 0,
    do_sample: bool = False,
    early_stopping: bool = False,
    num_beams: int = 1,
    num_beam_groups: int = 1,
    diversity_penalty: float = 0.0,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    typical_p: float = 1.0,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    encoder_no_repeat_ngram_size: int = 0,
    bad_words_ids: Optional[List[List[int]]] = None,
    num_return_sequences: int = 1,
    output_scores: bool = False,
    return_dict_in_generate: bool = False,
    forced_bos_token_id: Optional[int] = None,
    forced_eos_token_id: Optional[int] = None,
    remove_invalid_values: bool = False
    ) -> Union[List[str], dict]:
    """
    Generate text using a pre-trained transformer model.

    Args:
        model (PreTrainedModel): The pre-trained model to use for generation.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        prompts (Union[str, List[str]]): The input prompt(s) for text generation.
        (other arguments as described in the original list)

    Returns:

    """
    try:
        # Prepare inputs
        if isinstance(prompts, str):
            prompts = [prompts]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

        # Log generation start
        logger.info(f"Starting text generation for {len(prompts)} prompt(s)")

        # Generate text
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            early_stopping=early_stopping,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            typical_p=typical_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            num_return_sequences=num_return_sequences,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            remove_invalid_values=remove_invalid_values
        )

        # Decode generated text
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Log generation completion
        logger.info(f"Text generation completed successfully")

        if return_dict_in_generate:
            return {
                "generated_texts": generated_texts,
                "scores": outputs.scores if output_scores else None
            }
        else:
            return generated_texts

    except Exception as e:
        logger.error(f"An error occurred during text generation: {str(e)}")
        raise
