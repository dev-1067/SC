"""
Smart Traffic Signal Control System (STCS)
Person 1 Deliverable: Genetic Algorithm Optimizer (ga_optimizer.py)
Covers Syllabus Module 6:
- Real-Coded Chromosome Representation for Fuzzy MF Vertex Coordinates
- Selection Operators: Tournament Selection & Roulette-Wheel Selection
- Crossover Operator: Simulated Binary Crossover (SBX)
- Mutation Operator: Polynomial Mutation
- Elitism Preservation
- Multi-lane Dynamic Traffic Simulation Fitness Evaluator
- Parameter Export to JSON for Fuzzy Logic Controller
"""

import json
import os
import random
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from fuzzy_engine import FuzzyTrafficController


# ==============================================================================
# 1. CHROMOSOME ENCODING & DECODING
# ==============================================================================

class ChromosomeManager:
    """
    Manages translation between real-coded chromosome vector and fuzzy MF bounds.
    Maintains monotonicity constraints: a <= b <= c <= d for all MFs.
    """

    # Variable bounds and segment definitions: (var_name, term_name, mf_type, param_indices, min_val, max_val)
    GENE_SPECS = [
        # QueueLength [0, 50]
        ("QueueLength", "Short", "trap", [2, 3], 2.0, 22.0),
        ("QueueLength", "Medium", "tri", [0, 1, 2], 5.0, 35.0),
        ("QueueLength", "Long", "tri", [0, 1, 2], 18.0, 48.0),
        ("QueueLength", "VeryLong", "trap", [0, 1], 30.0, 50.0),

        # WaitingTime [0, 180]
        ("WaitingTime", "Low", "trap", [2, 3], 15.0, 80.0),
        ("WaitingTime", "Normal", "tri", [0, 1, 2], 30.0, 150.0),
        ("WaitingTime", "Critical", "trap", [0, 1], 90.0, 180.0),

        # ArrivalRate [0, 10]
        ("ArrivalRate", "Low", "trap", [2, 3], 1.0, 5.0),
        ("ArrivalRate", "Medium", "tri", [0, 1, 2], 1.5, 8.5),
        ("ArrivalRate", "High", "trap", [0, 1], 4.5, 10.0),

        # GreenExtension [10, 60]
        ("GreenExtension", "VeryShort", "trap", [2, 3], 11.0, 22.0),
        ("GreenExtension", "Short", "tri", [0, 1, 2], 13.0, 32.0),
        ("GreenExtension", "Medium", "tri", [0, 1, 2], 22.0, 46.0),
        ("GreenExtension", "Long", "tri", [0, 1, 2], 36.0, 56.0),
        ("GreenExtension", "VeryLong", "trap", [0, 1], 46.0, 60.0),
    ]

    def __init__(self):
        # Calculate total genes
        self.num_genes = sum(len(spec[3]) for spec in self.GENE_SPECS)
        self.bounds = self._build_bounds()

    def _build_bounds(self) -> List[Tuple[float, float]]:
        bounds = []
        for _, _, _, indices, min_v, max_v in self.GENE_SPECS:
            for _ in indices:
                bounds.append((min_v, max_v))
        return bounds

    def params_to_chromosome(self, params: Dict[str, Any]) -> np.ndarray:
        """Extract float array from nested parameter dict."""
        genes = []
        for var_name, term_name, _, indices, _, _ in self.GENE_SPECS:
            term_params = params[var_name][term_name]
            for idx in indices:
                genes.append(float(term_params[idx]))
        return np.array(genes, dtype=np.float64)

    def chromosome_to_params(self, chromosome: np.ndarray, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert chromosome array back to structured fuzzy parameter dictionary."""
        params_copy = json.loads(json.dumps(base_params))
        gene_idx = 0

        for var_name, term_name, mf_type, indices, min_v, max_v in self.GENE_SPECS:
            n_vals = len(indices)
            sub_genes = chromosome[gene_idx: gene_idx + n_vals]
            gene_idx += n_vals

            # Clip within physical bounds and sort to maintain fuzzy set monotonicity
            clipped_sorted = np.sort(np.clip(sub_genes, min_v, max_v))
            target_params = params_copy[var_name][term_name]

            for i, p_idx in enumerate(indices):
                target_params[p_idx] = round(float(clipped_sorted[i]), 3)

            # Enforce overall parameter monotonicity
            if mf_type == "tri":
                target_params[:3] = sorted(target_params[:3])
            elif mf_type == "trap":
                target_params[:4] = sorted(target_params[:4])

        # Enforce complete Ruspini-style partition coverage across continuous universes
        # 1. QueueLength [0, 50]
        q = params_copy["QueueLength"]
        q["Medium"][0] = min(q["Medium"][0], q["Short"][2])
        q["Short"][3] = max(q["Short"][3], q["Medium"][1])
        q["Long"][0] = min(q["Long"][0], q["Medium"][1])
        q["Medium"][2] = max(q["Medium"][2], q["Long"][1])
        q["VeryLong"][0] = min(q["VeryLong"][0], q["Long"][1])
        q["Long"][2] = max(q["Long"][2], q["VeryLong"][1])

        # 2. WaitingTime [0, 180]
        w = params_copy["WaitingTime"]
        w["Normal"][0] = min(w["Normal"][0], w["Low"][2])
        w["Low"][3] = max(w["Low"][3], w["Normal"][1])
        w["Critical"][0] = min(w["Critical"][0], w["Normal"][1])
        w["Normal"][2] = max(w["Normal"][2], w["Critical"][1])

        # 3. ArrivalRate [0, 10]
        arr = params_copy["ArrivalRate"]
        arr["Medium"][0] = min(arr["Medium"][0], arr["Low"][2])
        arr["Low"][3] = max(arr["Low"][3], arr["Medium"][1])
        arr["High"][0] = min(arr["High"][0], arr["Medium"][1])
        arr["Medium"][2] = max(arr["Medium"][2], arr["High"][1])

        # 4. GreenExtension [10, 60]
        ge = params_copy["GreenExtension"]
        ge["Short"][0] = min(ge["Short"][0], ge["VeryShort"][2])
        ge["VeryShort"][3] = max(ge["VeryShort"][3], ge["Short"][1])
        ge["Medium"][0] = min(ge["Medium"][0], ge["Short"][1])
        ge["Short"][2] = max(ge["Short"][2], ge["Medium"][1])
        ge["Long"][0] = min(ge["Long"][0], ge["Medium"][1])
        ge["Medium"][2] = max(ge["Medium"][2], ge["Long"][1])
        ge["VeryLong"][0] = min(ge["VeryLong"][0], ge["Long"][1])
        ge["Long"][2] = max(ge["Long"][2], ge["VeryLong"][1])

        # Final monotonic sort per term
        for v_dict in [q, w, arr, ge]:
            for t_k in v_dict:
                v_dict[t_k] = [round(float(val), 3) for val in sorted(v_dict[t_k])]

        return params_copy

    def create_random_chromosome(self) -> np.ndarray:
        """Sample a valid random chromosome uniformly within bounds."""
        genes = []
        for _, _, _, indices, min_v, max_v in self.GENE_SPECS:
            n_vals = len(indices)
            raw = np.sort(np.random.uniform(min_v, max_v, size=n_vals))
            genes.extend(raw)
        return np.array(genes, dtype=np.float64)


# ==============================================================================
# 2. TRAFFIC SIMULATOR & FITNESS EVALUATOR (Module 6 Specification)
# ==============================================================================

class TrafficSimulationEvaluator:
    """
    Evaluates candidate chromosome fitness via an iterative 4-way intersection simulation.
    Objective:
        Minimize Fitness F = Sum_t ( Sum_lanes Queue_i(t) * WaitTime_i(t) )
    """
    def __init__(self, simulation_cycles: int = 50, random_seed: int = 42):
        self.simulation_cycles = simulation_cycles
        self.random_seed = random_seed
        self._precompute_demands()

    def _precompute_demands(self):
        """Generate deterministic synthetic arrival patterns across simulation horizon."""
        np.random.seed(self.random_seed)
        self.demands = []
        # Phases: 0 = North-South, 1 = East-West
        for c in range(self.simulation_cycles):
            # Surge pattern: surge in NS, then EW, then balanced
            if c < 15:
                arr_ns = np.random.uniform(6.0, 9.5)
                arr_ew = np.random.uniform(1.5, 4.0)
            elif c < 30:
                arr_ns = np.random.uniform(2.0, 4.5)
                arr_ew = np.random.uniform(6.5, 9.5)
            else:
                arr_ns = np.random.uniform(4.0, 7.5)
                arr_ew = np.random.uniform(4.0, 7.5)
            self.demands.append({"NS": arr_ns, "EW": arr_ew})

    def evaluate_fitness(self, controller: FuzzyTrafficController) -> float:
        """
        Runs the simulation with given controller parameters and computes cumulative delay penalty.
        """
        # State: queues and wait times for Phase 0 (NS) and Phase 1 (EW)
        state = {
            "NS": {"queue": 15.0, "wait": 20.0},
            "EW": {"queue": 12.0, "wait": 25.0}
        }
        active_phase = "NS"
        total_penalty = 0.0
        discharge_rate = 0.8  # vehicles discharged per second of green light

        for c, demand in enumerate(self.demands):
            # Inflow during cycle interval
            arr_active = demand[active_phase]
            other_phase = "EW" if active_phase == "NS" else "NS"
            arr_other = demand[other_phase]

            # FLC determines green extension for active phase
            flc_res = controller.calculate_green_extension(
                queue_length=state[active_phase]["queue"],
                waiting_time=state[active_phase]["wait"],
                arrival_rate=arr_active
            )
            green_duration = flc_res["green_extension_seconds"]

            # Vehicle discharge on active phase
            discharged = green_duration * discharge_rate
            state[active_phase]["queue"] = max(0.0, state[active_phase]["queue"] + (arr_active * green_duration / 60.0) - discharged)
            state[active_phase]["wait"] = max(0.0, state[active_phase]["wait"] * 0.3)  # Wait drops as flow clears

            # Accumulation on stopped competitor phase
            state[other_phase]["queue"] += arr_other * green_duration / 60.0
            state[other_phase]["wait"] += green_duration

            # Calculate penalty for this cycle: sum(Queue_i * WaitTime_i)
            cycle_penalty = (state["NS"]["queue"] * state["NS"]["wait"]) + (state["EW"]["queue"] * state["EW"]["wait"])
            total_penalty += cycle_penalty

            # Phase transition logic
            if state[other_phase]["queue"] > state[active_phase]["queue"] or state[other_phase]["wait"] > 90.0:
                active_phase = other_phase

        return float(total_penalty)


# ==============================================================================
# 3. GENETIC ALGORITHM OPERATORS & SOLVER (Module 6)
# ==============================================================================

class Individual:
    """Represents a candidate solution in the GA population."""
    def __init__(self, chromosome: np.ndarray):
        self.chromosome = np.array(chromosome, dtype=np.float64)
        self.fitness: float = float("inf")  # Lower is better (minimization)


class GAOptimizer:
    """
    Real-Coded Genetic Algorithm for Auto-Tuning Fuzzy Controller Membership Parameters.
    Features:
    - Simulated Binary Crossover (SBX)
    - Polynomial Mutation
    - Tournament & Roulette-Wheel Selection
    - Elitism
    - Progress Monitoring & Convergence History
    """
    def __init__(
        self,
        population_size: int = 30,
        generations: int = 25,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.15,
        tournament_size: int = 3,
        elite_count: int = 2,
        eta_c: float = 2.0,   # SBX distribution index
        eta_m: float = 20.0,  # Polynomial mutation distribution index
        random_seed: int = 42,
        verbose: bool = True
    ):
        self.pop_size = population_size
        self.generations = generations
        self.p_cross = crossover_rate
        self.p_mut = mutation_rate
        self.tourn_size = tournament_size
        self.elite_count = elite_count
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.verbose = verbose

        np.random.seed(random_seed)
        random.seed(random_seed)

        self.chrom_manager = ChromosomeManager()
        self.evaluator = TrafficSimulationEvaluator(simulation_cycles=40, random_seed=random_seed)
        self.base_controller = FuzzyTrafficController()
        self.base_params = self.base_controller.get_parameters()

        # Convergence tracking
        self.history: List[Dict[str, float]] = []
        self.best_individual: Optional[Individual] = None

    def _evaluate(self, ind: Individual):
        """Evaluate individual fitness using simulation."""
        params = self.chrom_manager.chromosome_to_params(ind.chromosome, self.base_params)
        eval_controller = FuzzyTrafficController(params)
        ind.fitness = self.evaluator.evaluate_fitness(eval_controller)

    # --------------------------------------------------------------------------
    # SELECTION OPERATORS
    # --------------------------------------------------------------------------
    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Select best candidate among randomly chosen subset."""
        candidates = random.sample(population, self.tourn_size)
        return min(candidates, key=lambda ind: ind.fitness)

    def roulette_wheel_selection(self, population: List[Individual]) -> Individual:
        """Fitness-proportionate selection (inverse rank for minimization)."""
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)
        ranks = np.arange(len(sorted_pop), 0, -1, dtype=float)
        probs = ranks / ranks.sum()
        chosen_idx = np.random.choice(len(sorted_pop), p=probs)
        return sorted_pop[chosen_idx]

    # --------------------------------------------------------------------------
    # CROSSOVER OPERATOR: Simulated Binary Crossover (SBX)
    # --------------------------------------------------------------------------
    def simulated_binary_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Real-coded Simulated Binary Crossover (SBX)."""
        if np.random.rand() > self.p_cross:
            return parent1.copy(), parent2.copy()

        c1 = parent1.copy()
        c2 = parent2.copy()

        for i in range(len(parent1)):
            if np.random.rand() <= 0.5:
                p1 = parent1[i]
                p2 = parent2[i]
                if abs(p1 - p2) > 1e-6:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2.0 * u) ** (1.0 / (self.eta_c + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta_c + 1.0))
                    
                    c1[i] = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
                    c2[i] = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)

        return c1, c2

    # --------------------------------------------------------------------------
    # MUTATION OPERATOR: Polynomial Mutation
    # --------------------------------------------------------------------------
    def polynomial_mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """Real-coded Polynomial Mutation preserving variable bounds."""
        mutated = chromosome.copy()
        bounds = self.chrom_manager.bounds

        for i in range(len(chromosome)):
            if np.random.rand() < self.p_mut:
                lb, ub = bounds[i]
                val = mutated[i]
                delta1 = (val - lb) / (ub - lb + 1e-8)
                delta2 = (ub - val) / (ub - lb + 1e-8)
                
                u = np.random.rand()
                mut_pow = 1.0 / (self.eta_m + 1.0)
                
                if u <= 0.5:
                    xy = 1.0 - delta1
                    val_factor = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (self.eta_m + 1.0))
                    delta_q = (val_factor ** mut_pow) - 1.0
                else:
                    xy = 1.0 - delta2
                    val_factor = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (self.eta_m + 1.0))
                    delta_q = 1.0 - (val_factor ** mut_pow)

                mutated[i] = np.clip(val + delta_q * (ub - lb), lb, ub)

        return mutated

    # --------------------------------------------------------------------------
    # MAIN OPTIMIZATION LOOP
    # --------------------------------------------------------------------------
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Executes full Genetic Algorithm tuning pipeline."""
        if verbose:
            print("=" * 70)
            print(f"Starting GA Optimization for Fuzzy Traffic Signal Controller")
            print(f"Population: {self.pop_size} | Generations: {self.generations} | Genes: {self.chrom_manager.num_genes}")
            print("=" * 70)

        # Baseline evaluation
        base_ind = Individual(self.chrom_manager.params_to_chromosome(self.base_params))
        self._evaluate(base_ind)
        baseline_fitness = base_ind.fitness
        if verbose:
            print(f"Initial Unoptimized Baseline Penalty F = {baseline_fitness:.2f}")

        # Initialize population
        population: List[Individual] = [base_ind]  # Seed with baseline
        for _ in range(self.pop_size - 1):
            chrom = self.chrom_manager.create_random_chromosome()
            ind = Individual(chrom)
            self._evaluate(ind)
            population.append(ind)

        population.sort(key=lambda ind: ind.fitness)
        self.best_individual = population[0]

        for gen in range(1, self.generations + 1):
            new_population: List[Individual] = []

            # 1. Elitism: preserve top elite_count individuals
            for elite in population[:self.elite_count]:
                new_population.append(Individual(elite.chromosome.copy()))

            # 2. Reproduction loop
            while len(new_population) < self.pop_size:
                p1 = self.tournament_selection(population)
                p2 = self.tournament_selection(population)

                c1_chrom, c2_chrom = self.simulated_binary_crossover(p1.chromosome, p2.chromosome)
                c1_chrom = self.polynomial_mutation(c1_chrom)
                c2_chrom = self.polynomial_mutation(c2_chrom)

                new_population.append(Individual(c1_chrom))
                if len(new_population) < self.pop_size:
                    new_population.append(Individual(c2_chrom))

            # 3. Evaluate offspring
            for ind in new_population:
                self._evaluate(ind)

            new_population.sort(key=lambda ind: ind.fitness)
            population = new_population

            if population[0].fitness < self.best_individual.fitness:
                self.best_individual = population[0]

            fits = [ind.fitness for ind in population]
            gen_stat = {
                "generation": gen,
                "best_fitness": round(float(np.min(fits)), 2),
                "avg_fitness": round(float(np.mean(fits)), 2),
                "worst_fitness": round(float(np.max(fits)), 2)
            }
            self.history.append(gen_stat)

            if verbose and (gen % 5 == 0 or gen == 1 or gen == self.generations):
                print(f"Gen {gen:02d}/{self.generations:02d} | Best: {gen_stat['best_fitness']:10.2f} | "
                      f"Avg: {gen_stat['avg_fitness']:10.2f} | Improvement: {((baseline_fitness - gen_stat['best_fitness'])/baseline_fitness)*100:5.2f}%")

        # Compile results
        best_params = self.chrom_manager.chromosome_to_params(self.best_individual.chromosome, self.base_params)
        improvement_pct = ((baseline_fitness - self.best_individual.fitness) / baseline_fitness) * 100.0

        if verbose:
            print("=" * 70)
            print(f"Optimization Complete!")
            print(f"Baseline Penalty: {baseline_fitness:.2f} -> Optimized Penalty: {self.best_individual.fitness:.2f}")
            print(f"Delay / Spillback Reduction: {improvement_pct:.2f}% (Target: >= 20-25%)")
            print("=" * 70)

        return {
            "baseline_fitness": baseline_fitness,
            "optimized_fitness": self.best_individual.fitness,
            "improvement_percent": round(improvement_pct, 2),
            "optimized_params": best_params,
            "convergence_history": self.history
        }

    def save_optimized_parameters(self, output_path: str = "optimized_fuzzy_params.json") -> str:
        """Serialize best fuzzy parameters to JSON."""
        if not self.best_individual:
            raise RuntimeError("Run GA optimization before saving parameters.")
        
        best_params = self.chrom_manager.chromosome_to_params(self.best_individual.chromosome, self.base_params)
        with open(output_path, "w") as f:
            json.dump(best_params, f, indent=2)
        return output_path


# ==============================================================================
# SCRIPT ENTRYPOINT FOR STANDALONE EXECUTION
# ==============================================================================

if __name__ == "__main__":
    opt = GAOptimizer(population_size=20, generations=15, verbose=True)
    res = opt.run(verbose=True)
    save_file = opt.save_optimized_parameters()
    print(f"Successfully saved tuned fuzzy parameters to: {save_file}")
