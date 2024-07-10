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

# Define the fitness function
def evaluate_perturbation(perturbation: List[int], model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: str) -> float:
    inputs = torch.tensor(perturbation, device=device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(inputs)[0]
    return outputs.sum().item()

def create_population(size: int, dimension: int) -> List[List[int]]:
    return [random.choices(range(dimension), k=dimension) for _ in range(size)]

def mutate_individual(individual: List[int], mutation_prob: float, dimension: int) -> None:
    for i in range(len(individual)):
        if random.random() < mutation_prob:
            individual[i] = random.randint(0, dimension-1)

def crossover_individuals(ind1: List[int], ind2: List[int], crossover_prob: float) -> Tuple[List[int], List[int]]:
    if random.random() < crossover_prob:
        point = random.randint(1, len(ind1) - 1)
        return ind1[:point] + ind2[point:], ind2[:point] + ind1[point:]
    return ind1, ind2

def main(
    target_model: PreTrainedModel,
    target_tokenizer: PreTrainedTokenizer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    population_size: int = 50,
    generations: int = 100,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.2,
    metric_name: str = "accuracy"
) -> None:

    # Define custom individual, fitness, and toolbox for DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, target_tokenizer.vocab_size - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=10)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_perturbation, model=target_model, tokenizer=target_tokenizer, device=device)
    toolbox.register("mate", crossover_individuals, crossover_prob=crossover_prob)
    toolbox.register("mutate", mutate_individual, mutation_prob=mutation_prob, dimension=target_tokenizer.vocab_size)
    toolbox.register("select", tools.selBest)

    population = toolbox.population(n=population_size)

    try:
        for gen in range(generations):
            logger.info(f"Generation {gen}")

            # Evaluate individuals
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = (fit,)

            # Select the next generation
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                toolbox.mate(child1, child2)
                toolbox.mutate(child1)
                toolbox.mutate(child2)
                del child1.fitness.values
                del child2.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit,)

            # Replace population
            population[:] = offspring

        best_ind = tools.selBest(population, 1)[0]
        logger.info(f"Best perturbation: {best_ind}")
        logger.info(f"Best fitness: {best_ind.fitness.values}")
        
    except Exception as e:
        logger.error(f"Error during evolutionary algorithm: {e}")
        logger.info("Best perturbation: []")
        logger.info("Best fitness: None")
        



class GradientEstimationAttack:
    def __init__(
        self,
        target_model: PreTrainedModel,
        target_tokenizer: PreTrainedTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        population_size: int = 50,
        generations: int = 100,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        metric_name: str = "accuracy",
        input_text: str = "Sample text for attack"
    ):
        self.target_model = target_model.to(device)
        self.target_tokenizer = target_tokenizer
        self.device = device
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.metric_name = metric_name
        self.metric = evaluate.load(metric_name)
        self.input_text = input_text

        self.input_ids = self.target_tokenizer.encode(self.input_text, return_tensors="pt").to(self.device)
        self.seq_length = self.input_ids.shape[1]

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", random.uniform, -1, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attribute, n=self.seq_length)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.fitness_function)

    def estimate_gradient(self, input_ids: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """Estimate gradients using finite differences method."""
        original_output = self.target_model(input_ids).logits
        estimated_gradient = torch.zeros_like(input_ids, dtype=torch.float)

        for i in range(input_ids.shape[1]):
            perturbed_input = input_ids.clone()
            perturbed_input[0, i] += epsilon
            perturbed_output = self.target_model(perturbed_input).logits
            estimated_gradient[0, i] = (perturbed_output - original_output).sum() / epsilon

        return estimated_gradient

    def fitness_function(self, individual: List[float]) -> Tuple[float]:
        """Evaluate the fitness of an individual."""
        perturbation = torch.tensor(individual, device=self.device).unsqueeze(0)
        perturbed_input_ids = self.input_ids.clone()
        perturbed_input_ids += perturbation.long()  # Convert to long to match input_ids dtype
        
        with torch.no_grad():
            output = self.target_model(perturbed_input_ids).logits
        
        predictions = torch.argmax(output, dim=-1).squeeze().cpu().tolist()
        reference = torch.argmax(self.target_model(self.input_ids).logits, dim=-1).squeeze().cpu().tolist()
        
        metric_result = self.metric.compute(predictions=predictions, references=reference)
        return (metric_result[self.metric_name],)

    def run_attack(self) -> Tuple[List[float], Any]:
        """Run the gradient estimation attack."""
        population = self.toolbox.population(n=self.population_size)
        
        try:
            algorithms.eaSimple(
                population,
                self.toolbox,
                cxpb=self.crossover_prob,
                mutpb=self.mutation_prob,
                ngen=self.generations,
                stats=None,
                halloffame=None,
                verbose=True
            )
        except Exception as e:
            logger.error(f"Error during evolutionary algorithm: {str(e)}")
            return [], None

        best_individual = tools.selBest(population, k=1)[0]
        return best_individual, best_individual.fitness.values[0]



if __name__ == "__main__":
    # Example usage
    model_name = "gpt2"
    cache_dir=r"C:\Users\heman\Desktop\Coding\data"
    model = AutoModelForCausalLM.from_pretrained(model_name,cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    main(
        target_model=model,
        target_tokenizer=tokenizer,
        population_size=50,
        generations=100,
        crossover_prob=0.7,
        mutation_prob=0.2,
        metric_name="accuracy"
    )

    # attack = GradientEstimationAttack(model, tokenizer)
    # best_perturbation, best_fitness = attack.run_attack()

    # logger.info(f"Best perturbation: {best_perturbation}")
    # logger.info(f"Best fitness: {best_fitness}")