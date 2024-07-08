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
    EvalPrediction,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer
)
import evaluate
from deap import base, creator, tools, algorithms

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




import torch
import random
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)
import torch
import random
from typing import List, Tuple, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DecisionBasedAttack:
    def __init__(
        self,
        target_model: PreTrainedModel,
        target_tokenizer: PreTrainedTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        population_size: int = 50,
        generations: int = 100,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        max_length: int = 50
    ):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.device = device
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_length = max_length
        
        self.target_model.to(self.device)
        self.target_model.eval()
        
        # Get valid token IDs (excluding special tokens)
        self.valid_token_ids = [id for id in range(self.target_tokenizer.vocab_size) 
                                if id not in self.target_tokenizer.all_special_ids]
        
        # Set up DEAP tools
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", random.choice, self.valid_token_ids)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attribute, n=self.max_length)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def mutate_individual(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mutation_prob:
                individual[i] = random.choice(self.valid_token_ids)
        return (individual,)

    def evaluate_individual(self, individual: List[int]) -> Tuple[float,]:
        try:
            tokens = torch.tensor(individual).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.target_model(tokens)
            
            # Use the mean of the maximum logit values as the fitness
            max_logits = output.logits.max(dim=-1).values
            fitness = max_logits.mean().item()
            
            return (fitness,)
        except Exception as e:
            logger.error(f"Error in evaluate_individual: {str(e)}")
            return (0.0,)

    def attack(self, input_text: str) -> Optional[str]:
        try:
            logger.info(f"Starting attack on input: {input_text}")
            
            input_tokens = self.target_tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            
            population = self.toolbox.population(n=self.population_size)
            
            for gen in range(self.generations):
                offspring = algorithms.varAnd(population, self.toolbox, cxpb=self.crossover_prob, mutpb=self.mutation_prob)
                fits = self.toolbox.map(self.toolbox.evaluate, offspring)
                for fit, ind in zip(fits, offspring):
                    ind.fitness.values = fit
                population = self.toolbox.select(offspring, k=len(population))
                
                best_ind = tools.selBest(population, k=1)[0]
                logger.info(f"Generation {gen}: Best fitness = {best_ind.fitness.values[0]}")
            
            best_solution = tools.selBest(population, k=1)[0]
            adversarial_text = self.target_tokenizer.decode(best_solution, skip_special_tokens=True)
            
            logger.info(f"Attack completed. Adversarial text: {adversarial_text}")
            
            return adversarial_text
        except Exception as e:
            logger.error(f"Error in attack: {str(e)}")
            return None
def main():
    try:
        # Load model and tokenizer
        model_name = "gpt2"
        cache_dir = r"E:\LLMS\Fine-tuning\llms-data"
        logger.info(f"Loading model and tokenizer from {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Model and tokenizer loaded successfully")

        # Initialize attack
        attack = DecisionBasedAttack(model, tokenizer)
        logger.info("DecisionBasedAttack initialized")

        # Get user input
        input_text = "what is the meaning of life??"
        logger.info(f"Input text: {input_text}")

        # Perform attack
        adversarial_text = attack.attack(input_text)

        if adversarial_text:
            logger.info(f"Original text: {input_text}")
            logger.info(f"Adversarial text: {adversarial_text}")
            print(f"Original text: {input_text}")
            print(f"Adversarial text: {adversarial_text}")
        else:
            logger.warning("Attack failed to produce adversarial text.")
            print("Attack failed to produce adversarial text.")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()