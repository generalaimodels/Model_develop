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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvolutionaryAttack:
    def __init__(
        self,
        target_model: PreTrainedModel,
        target_tokenizer: PreTrainedTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        population_size: int = 50,
        generations: int = 100,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        metric_name: str = "accuracy"
    ):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.device = device
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.metric_name = metric_name

        self.metric = evaluate.load(metric_name)
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def create_individual(self) -> creator.Individual:
        """Create a random individual (sequence of tokens)."""
        try:
            length = random.randint(10, 50)
            return creator.Individual(random.choices(list(self.target_tokenizer.get_vocab().keys()), k=length))
        except Exception as e:
            logger.error(f"Error creating individual: {str(e)}")
            raise

    def mutate_individual(self, individual: creator.Individual) -> Tuple[creator.Individual]:
        """Mutate an individual by randomly replacing tokens."""
        try:
            for i in range(len(individual)):
                if random.random() < self.mutation_prob:
                    individual[i] = random.choice(list(self.target_tokenizer.get_vocab().keys()))
            return (individual,)
        except Exception as e:
            logger.error(f"Error mutating individual: {str(e)}")
            raise

    def evaluate_individual(self, individual: creator.Individual) -> Tuple[float]:
        """Evaluate the fitness of an individual."""
        try:
            text = self.target_tokenizer.decode(individual)
            inputs = self.target_tokenizer(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.target_model(**inputs)
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            metric_result = self.metric.compute(predictions=predictions.cpu().numpy(), 
                                                references=inputs["input_ids"].cpu().numpy())
            
            return (metric_result[self.metric_name],)
        except Exception as e:
            logger.error(f"Error evaluating individual: {str(e)}")
            raise

    def run_attack(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Run the evolutionary attack."""
        try:
            population = self.toolbox.population(n=self.population_size)
            
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", torch.mean)
            stats.register("std", torch.std)
            stats.register("min", torch.min)
            stats.register("max", torch.max)
            
            logger.info("Starting evolutionary attack...")
            population, logbook = algorithms.eaSimple(population, self.toolbox, 
                                                      cxpb=self.crossover_prob, 
                                                      mutpb=self.mutation_prob, 
                                                      ngen=self.generations, 
                                                      stats=stats, 
                                                      verbose=True)
            
            best_individual = tools.selBest(population, k=1)[0]
            logger.info(f"Best individual: {self.target_tokenizer.decode(best_individual)}")
            logger.info(f"Best fitness: {best_individual.fitness.values[0]}")
            
            return population, logbook
        except Exception as e:
            logger.error(f"Error running attack: {str(e)}")
            raise

def main():
    try:
        model_name = "gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        attack = EvolutionaryAttack(model, tokenizer)
        population, logbook = attack.run_attack()
        logger.info("Attack completed successfully.")
        logger.info(f"Final population size: {len(population)}")
        logger.info(f"Number of generations: {len(logbook)}")

    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main()