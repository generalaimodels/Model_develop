
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeneticAttack:
    def __init__(
        self,
        target_model: PreTrainedModel,
        target_tokenizer: PreTrainedTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        population_size: int = 50,
        generations: int = 100,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        sequence_length: int = 50
    ):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.device = device
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.sequence_length = sequence_length
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", random.randint, 0, self.target_tokenizer.vocab_size - 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attribute, n=self.sequence_length)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutUniformInt, low=0, up=self.target_tokenizer.vocab_size - 1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate_individual(self, individual: List[int]) -> Tuple[float]:
        try:
            input_ids = torch.tensor(individual).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.target_model(input_ids)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # Calculate a custom metric (e.g., difference between input and output)
            difference = torch.sum(torch.abs(predictions - input_ids)).item()
            similarity = 1 / (1 + difference)  # Convert to a similarity score
            return (similarity,)
        except Exception as e:
            logger.error(f"Error during individual evaluation: {str(e)}")
            return (0.0,)

    def run_attack(self) -> Tuple[List[int], float]:
        try:
            population = self.toolbox.population(n=self.population_size)
            
            for gen in range(self.generations):
                offspring = algorithms.varAnd(population, self.toolbox, cxpb=self.crossover_prob, mutpb=self.mutation_prob)
                fits = self.toolbox.map(self.toolbox.evaluate, offspring)
                for fit, ind in zip(fits, offspring):
                    ind.fitness.values = fit
                population = self.toolbox.select(offspring, k=len(population))
                
                best_ind = tools.selBest(population, k=1)[0]
                logger.info(f"Generation {gen}: Best fitness = {best_ind.fitness.values[0]}")
            
            best_individual = tools.selBest(population, k=1)[0]
            return best_individual, best_individual.fitness.values[0]
        except Exception as e:
            logger.error(f"Error during attack execution: {str(e)}")
            return [], 0.0

def main():
    try:
        model_name = "gpt2"
        cache_dir = r"C:\Users\heman\Desktop\Coding\data"
        logger.info(f"Loading model and tokenizer from {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize and run the genetic attack
        attack = GeneticAttack(model, tokenizer)
        best_individual, best_fitness = attack.run_attack()
        
        logger.info(f"Best individual: {best_individual}")
        logger.info(f"Best fitness: {best_fitness}")
        
        # Decode the best individual
        best_text = tokenizer.decode(best_individual)
        logger.info(f"Best text: {best_text}")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()

class AdversarialAttack:
    def __init__(self, target_model: PreTrainedModel, target_tokenizer: PreTrainedTokenizer, device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 population_size: int = 50, generations: int = 100, crossover_prob: float = 0.7, mutation_prob: float = 0.2, metric_name: str = "accuracy") -> None:
        self.model = target_model.to(device)
        self.tokenizer = target_tokenizer
        self.device = device
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.metric_name = metric_name
        self.metric = evaluate.load(metric_name)

        self.toolbox = base.Toolbox()
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    def evaluate_individual(self, individual: List[int]) -> Tuple[float]:
        text = self.tokenizer.decode(individual, skip_special_tokens=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        
        metric_result = self.metric.compute(predictions=preds, references=inputs["input_ids"])
        return metric_result.get(self.metric_name, 0.0),

    def mutate_individual(self, individual: List[int]) -> Tuple[List[int]]:
        idx = random.randrange(len(individual))
        individual[idx] = self.tokenizer.vocab_size
        return individual,

    def run(self) -> Any:
        logger.info("Initializing population")
        self.toolbox.register("attr_int", random.randint, 0, self.tokenizer.vocab_size - 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_int, n=10)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        population = self.toolbox.population(n=self.population_size)
        logger.info("Starting genetic algorithm")
        
        algorithms.eaSimple(population, self.toolbox, cxpb=self.crossover_prob, mutpb=self.mutation_prob, ngen=self.generations,
                            stats=None, halloffame=None, verbose=True)

        top_individual = tools.selBest(population, k=1)[0]
        logger.info(f"Best individual: {top_individual}")
        return top_individual

# def main():
#     try:
#         model_name = "gpt2"
#         cache_dir = r"C:\Users\heman\Desktop\Coding\data"
#         logger.info(f"Loading model and tokenizer from {model_name}")
#         model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
#         tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
#         tokenizer.pad_token = tokenizer.eos_token

#         attack = AdversarialAttack(
#             target_model=model,
#             target_tokenizer=tokenizer,
#             population_size=50,
#             generations=100,
#             crossover_prob=0.7,
#             mutation_prob=0.2,
#             metric_name="accuracy"
#         )
#         best_adversarial_input = attack.run()
#         logger.info(f"Best adversarial input: {best_adversarial_input}")

#     except Exception as e:
#         logger.error(f"An error occurred: {str(e)}", exc_info=True)

# if __name__ == "__main__":
#     main()
# def main():
#     try:
#         model_name = "gpt2"
#         cache_dir = r"C:\Users\heman\Desktop\Coding\data"
#         logger.info(f"Loading model and tokenizer from {model_name}")
#         model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
#         tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
#         tokenizer.pad_token = tokenizer.eos_token
#         # # Load model and tokenizer
#         # model_name = "gpt2"
#         # tokenizer = AutoTokenizer.from_pretrained(model_name)
#         # model = AutoModelForCausalLM.from_pretrained(model_name)
        
#         # Initialize and run the genetic attack
#         attack = GeneticAttack(model, tokenizer)
#         best_individual, best_fitness = attack.run_attack()
        
#         logger.info(f"Best individual: {best_individual}")
#         logger.info(f"Best fitness: {best_fitness}")
        
#         # Decode the best individual
#         best_text = tokenizer.decode(best_individual)
#         logger.info(f"Best text: {best_text}")
#     except Exception as e:
#         logger.error(f"Error in main function: {str(e)}")

# if __name__ == "__main__":
#     main()