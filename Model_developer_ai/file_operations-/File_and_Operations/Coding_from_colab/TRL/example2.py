from typing import List, Dict

import torch
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLMWithValueHead
from datasets import load_dataset

from trl.core import LengthSampler


# Define constants
DEVICE = 0 if torch.cuda.is_available() else "cpu"

REF_MODEL_NAME = "lvwerra/gpt2-imdb"
MODEL_NAME = "lvwerra/gpt2-imdb-pos-v2"
REWARD_MODEL = "lvwerra/distilbert-imdb"
N_BEST_OF = 4


def build_dataset(
    tokenizer: AutoTokenizer,
    dataset_name: str = "imdb",
    input_min_text_length: int = 2,
    input_max_text_length: int = 8,
) -> datasets.Dataset:
    """
    Builds and preprocesses the dataset.
    """
    # Load and filter dataset
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    # Define input length sampler
    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """
        Tokenizes a sample from the dataset.
        """
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    # Tokenize and format dataset
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

def generate_best_of_responses(
    model: AutoModelForCausalLMWithValueHead,
    query_tensors: List[torch.Tensor],
    gen_kwargs: Dict[str, any],
    output_length_sampler: LengthSampler,
    n_best_of: int,
) -> List[List[str]]:
    """
    Generates multiple responses for each query and selects the best ones based on scores.
    """
    response_tensors_best_of = []
    for query in query_tensors:
        # Generate multiple responses
        outputs = model.generate(
            query.unsqueeze(0).to(DEVICE),
            num_return_sequences=n_best_of,
            **gen_kwargs,
        )

        # Decode and store responses
        responses = []
        for output in outputs:
            output = output[: output_length_sampler()]
            responses.append(tokenizer.decode(output))
        response_tensors_best_of.append(responses)
    return response_tensors_best_of


def calculate_best_of_scores(
    reward_pipe: pipeline,
    response_tensors_best_of: List[List[str]],
    sent_kwargs: Dict[str, any],
) -> List[float]:
    """
    Calculates scores for the best-of responses and selects the highest score for each query.
    """
    scores_best_of = []
    for responses in response_tensors_best_of:
        scores = torch.tensor([output[0]["score"] for output in reward_pipe(responses, **sent_kwargs)])
        scores_best_of.append(scores.max().item())
    return scores_best_of



def generate_responses(
    model: AutoModelForCausalLMWithValueHead,
    query_tensors: List[torch.Tensor],
    gen_kwargs: Dict[str, any],
    output_length_sampler: LengthSampler,
) -> List[str]:
    """
    Generates responses using the provided model and parameters.
    """
    response_tensors = []
    for query in query_tensors:
        # Generate response
        output = model.generate(query.unsqueeze(0).to(DEVICE), **gen_kwargs)[0]
        output = output[: output_length_sampler()]

        # Decode and store response
        response_tensors.append(tokenizer.decode(output))
    return response_tensors


def calculate_scores(
    reward_pipe: pipeline,
    response_tensors: List[str],
    sent_kwargs: Dict[str, any],
) -> List[float]:
    """
    Calculates sentiment scores for the generated responses.
    """
    scores = [output[0]["score"] for output in reward_pipe(response_tensors, **sent_kwargs)]
    return scores


def main():
    """
    Main function for running the script.
    """
    # Load models and tokenizer
    model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(REF_MODEL_NAME)
    reward_pipe = pipeline("sentiment-analysis", model=REWARD_MODEL, device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(REF_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Move models to GPU if available
    model.to(DEVICE)
    ref_model.to(DEVICE)

    # Build dataset
    dataset = build_dataset(tokenizer)

    # Define generation and scoring parameters
    gen_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}

    # Define output length sampler
    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    # Process a batch of data
    bs = 16
    output_data = dict()
    dataset.set_format("pandas")
    df_batch = dataset[:].sample(bs)
    output_data["query"] = df_batch["query"].tolist()
    query_tensors = df_batch["input_ids"].tolist()

    # Generate responses and calculate scores
    response_tensors_ref = generate_responses(ref_model, query_tensors, gen_kwargs, output_length_sampler)
    response_tensors = generate_responses(model, query_tensors, gen_kwargs, output_length_sampler)

    scores_ref = calculate_scores(reward_pipe, response_tensors_ref, sent_kwargs)
    scores = calculate_scores(reward_pipe, response_tensors, sent_kwargs)


# Generate and score best_of responses
    response_tensors_best_of = generate_best_of_responses(
    model, query_tensors, gen_kwargs, output_length_sampler, N_BEST_OF
)
    scores_best_of = calculate_best_of_scores(reward_pipe, response_tensors_best_of, sent_kwargs)

# Select the best response for each query based on scores
    best_responses = [
    response_tensors_best_of[i][a.argmax().item()] for i, a in enumerate(scores_best_of)
]

# Update output data with best_of results
    output_data["response (best_of)"] = best_responses
    output_data["scores (best_of)"] = scores_best_of 

    # ... (Code for generating and scoring best_of responses remains the same)

    # Store results in a dataframe
    df_results = pd.DataFrame(output_data)
    print(df_results)


if __name__ == "__main__":
    main()