import logging
from typing import List, Tuple, Optional, Any, Dict
import random
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from transformers import DataCollatorForLanguageModeling
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
import evaluate
from deap import base, creator, tools, algorithms

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridSubstituteEvolutionaryAttack:
    def __init__(
        self,
        target_model: PreTrainedModel,
        target_tokenizer: PreTrainedTokenizer,
        substitute_model: Optional[PreTrainedModel] = None,
        substitute_tokenizer: Optional[PreTrainedTokenizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        population_size: int = 50,
        generations: int = 100,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        metric_name: str = "accuracy"
    ):
        self.target_model = target_model.to(device)
        self.target_tokenizer = target_tokenizer
        self.substitute_model = substitute_model.to(device) if substitute_model else target_model
        self.substitute_tokenizer = substitute_tokenizer or target_tokenizer
        self.device = device

        # Evolutionary algorithm parameters
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        # Metrics
        self.metric = evaluate.load(metric_name)

        logger.info(f"Initialized HybridSubstituteEvolutionaryAttack with device: {device}")

    def collect_data(self, num_samples: int) -> Dataset:
        """Collect input-output pairs by querying the target model."""
        logger.info(f"Collecting {num_samples} samples from target model")
        data = []
        for _ in range(num_samples):
            input_text = self._generate_random_input()
            output = self._query_target_model(input_text)
            data.append({"text": input_text, "label": output})
        return Dataset.from_list(data)
    
    

    def train_substitute_model(self, dataset: Dataset) -> None:
        logger.info("Training substitute model")
        train_test = dataset.train_test_split(test_size=0.2)
        
        tokenized_datasets = train_test.map(
            self._tokenize_function,
            batched=True,
            remove_columns=train_test["train"].column_names
        )
    
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.substitute_tokenizer,
            mlm=False
        )
    
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            load_best_model_at_end=True,
        )
    
        trainer = Trainer(
            model=self.substitute_model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )
    
        trainer.train()
        eval_result = trainer.evaluate()
        logger.info(f"Substitute model evaluation: {eval_result}")


    def evolve_adversarial_examples(self, initial_population: List[str]) -> List[str]:
        """Evolve adversarial examples using a genetic algorithm."""
        logger.info("Evolving adversarial examples")
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, 
                         lambda: list(random.choice(initial_population)))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self._mutate_text)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self._fitness_function)

        population = toolbox.population(n=self.population_size)
        algorithms.eaSimple(population, toolbox, cxpb=self.crossover_prob, mutpb=self.mutation_prob,
                            ngen=self.generations, verbose=True)

        return [self._individual_to_text(ind) for ind in population]

    def evaluate_adversarial_examples(self, adversarial_examples: List[str]) -> float:
        """Evaluate the success rate of adversarial examples on the target model."""
        logger.info("Evaluating adversarial examples")
        successes = sum(self._query_target_model(ex) != self._query_substitute_model(ex) 
                        for ex in adversarial_examples)
        success_rate = successes / len(adversarial_examples)
        logger.info(f"Adversarial example success rate: {success_rate:.2f}")
        return success_rate

    def _generate_random_input(self) -> str:
        """Generate a random input text."""
        # This is a placeholder. In a real scenario, you might want to use a 
        # language model or a corpus to generate more realistic texts.
        words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        return " ".join(random.choices(words, k=random.randint(5, 10)))

    @torch.no_grad()
    def _query_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, input_text: str) -> int:
        """Query a model and return the output label."""
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        outputs = model(**inputs)
        
        # Get the logits for the last token
        last_token_logits = outputs.logits[:, -1, :]
        
        # Get the predicted token ID
        predicted_token_id = last_token_logits.argmax(dim=-1).item()
        
        return predicted_token_id

    def _query_target_model(self, input_text: str) -> int:
        """Query the target model and return the output label."""
        return self._query_model(self.target_model, self.target_tokenizer, input_text)

    def _query_substitute_model(self, input_text: str) -> int:
        """Query the substitute model and return the output label."""
        return self._query_model(self.substitute_model, self.substitute_tokenizer, input_text)

    def _mutate_text(self, individual: List[str], indpb: float = 0.1) -> Tuple[List[str]]:
        """Mutate the text by replacing words with synonyms or similar words."""
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = self._get_similar_word(individual[i])
        return individual,

    def _get_similar_word(self, word: str) -> str:
        """Get a similar word for mutation."""
        # This is a placeholder. In a real scenario, you might want to use
        # a word embedding model or a thesaurus to find similar words.
        return random.choice(["the", "a", "an", "this", "that", "these", "those"])

    def _individual_to_text(self, individual: List[str]) -> str:
        """Convert an individual (list of words) to a text string."""
        return " ".join(individual)

    def _fitness_function(self, individual: List[str]) -> Tuple[float]:
        """Fitness function for the evolutionary algorithm."""
        text = self._individual_to_text(individual)
        effectiveness = int(self._query_substitute_model(text) != self._query_target_model(text))
        imperceptibility = 1 - self._calculate_perturbation(text)
        return (effectiveness * imperceptibility,)

    def _calculate_perturbation(self, text: str) -> float:
        """Calculate the perturbation level of the adversarial example."""
        # This is a placeholder. In a real scenario, you might want to use
        # a more sophisticated method to calculate perturbation.
        original_text = self._generate_random_input()
        return 1 - (len(set(text.split()) & set(original_text.split())) / len(set(original_text.split())))
    
    
    def _tokenize_function(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Tokenize the input texts and prepare labels."""
        tokenized = self.substitute_tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    
    def _compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute metrics for model evaluation."""
        logits, labels = eval_pred
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        predictions = shift_logits.argmax(dim=-1)
        return self.metric.compute(predictions=predictions.flatten(), references=shift_labels.flatten())
    

# Testing code
if __name__ == "__main__":
    logger.info("Starting test of HybridSubstituteEvolutionaryAttack")
    cache_dir=r"E:\LLMS\Fine-tuning\llms-data"
    # Load models and tokenizers
    target_model = AutoModelForCausalLM.from_pretrained("gpt2",cache_dir=cache_dir)
    target_tokenizer = AutoTokenizer.from_pretrained("gpt2",cache_dir=cache_dir)
    substitute_model = AutoModelForCausalLM.from_pretrained("distilgpt2",cache_dir=cache_dir)
    substitute_tokenizer = AutoTokenizer.from_pretrained("distilgpt2",cache_dir=cache_dir)
    target_tokenizer.pad_token = target_tokenizer.eos_token
    substitute_tokenizer.pad_token = substitute_tokenizer.eos_token
    # Initialize attack
    attack = HybridSubstituteEvolutionaryAttack(
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        substitute_model=substitute_model,
        substitute_tokenizer=substitute_tokenizer
    )

    # Collect data
    dataset = attack.collect_data(num_samples=100)
    logger.info(f"Collected dataset with {len(dataset)} samples")

    # Train substitute model
    attack.train_substitute_model(dataset)

    # Generate initial population
    initial_population = [attack._generate_random_input() for _ in range(10)]

    # Evolve adversarial examples
    adversarial_examples = attack.evolve_adversarial_examples(initial_population)
    logger.info(f"Generated {len(adversarial_examples)} adversarial examples")

    # Evaluate adversarial examples
    success_rate = attack.evaluate_adversarial_examples(adversarial_examples)
    logger.info(f"Final success rate: {success_rate}")

    # Print some example adversarial texts
    logger.info("Example adversarial texts:")
    for i, text in enumerate(adversarial_examples[:5], 1):
        logger.info(f"Example {i}: {text}")

    logger.info("Test completed")