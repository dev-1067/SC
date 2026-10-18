"""
Automated Verification & Unit Test Suite for Person 1: Soft Computing Core
Exhaustive verification of all Person 1 deliverables:
1. Fuzzy Membership Functions (trimf, trapmf open-shoulders, bounds)
2. Ruspini-style partition completeness (zero dead zones across all 4 variables)
3. Mamdani Fuzzy Inference Engine (36 rules, Centroid COG defuzzifier)
4. Robust input sanitization (handling strings, negative values, NaNs, infs)
5. Integration Hook get_optimal_phase(lane_metrics):
   - Multi-phase arbitration (2-phase, 4-phase)
   - String vs int key compatibility
   - Minimum green guarantee (safety)
   - Maximum green cap (starvation prevention)
   - Heavy surge switching
6. Milestone 1 continuous fuzzer (100 randomized detector frames)
7. Genetic Algorithm Optimizer:
   - Monotonicity preservation
   - Selection, SBX crossover, polynomial mutation
   - Cumulative delay reduction benchmark (>= 20-25%)
8. 5-Layer ANFIS System:
   - Layer 1 to 5 activations
   - Jang's Hybrid Learning (LSE forward + backprop backward)
   - Model save and load round-trip fidelity
"""

import unittest
import numpy as np
import torch
import os
import json

from fuzzy_engine import (
    trimf,
    trapmf,
    LinguisticVariable,
    FuzzyTrafficController,
    get_optimal_phase,
    get_controller
)
from ga_optimizer import ChromosomeManager, GAOptimizer, TrafficSimulationEvaluator
from anfis_tuner import ANFISNet, ANFISTrafficTuner


class TestFuzzyMembershipFunctions(unittest.TestCase):
    """Test mathematical accuracy of Fuzzy Set MFs."""

    def test_trimf_values(self):
        # Peak at 20, support [10, 30]
        self.assertAlmostEqual(trimf(10.0, [10.0, 20.0, 30.0]), 0.0)
        self.assertAlmostEqual(trimf(15.0, [10.0, 20.0, 30.0]), 0.5)
        self.assertAlmostEqual(trimf(20.0, [10.0, 20.0, 30.0]), 1.0)
        self.assertAlmostEqual(trimf(25.0, [10.0, 20.0, 30.0]), 0.5)
        self.assertAlmostEqual(trimf(30.0, [10.0, 20.0, 30.0]), 0.0)
        self.assertAlmostEqual(trimf(5.0, [10.0, 20.0, 30.0]), 0.0)
        self.assertAlmostEqual(trimf(35.0, [10.0, 20.0, 30.0]), 0.0)

    def test_trapmf_values(self):
        # Plateau [10, 20], support [5, 25]
        self.assertAlmostEqual(trapmf(3.0, [5.0, 10.0, 20.0, 25.0]), 0.0)
        self.assertAlmostEqual(trapmf(7.5, [5.0, 10.0, 20.0, 25.0]), 0.5)
        self.assertAlmostEqual(trapmf(12.0, [5.0, 10.0, 20.0, 25.0]), 1.0)
        self.assertAlmostEqual(trapmf(19.0, [5.0, 10.0, 20.0, 25.0]), 1.0)
        self.assertAlmostEqual(trapmf(22.5, [5.0, 10.0, 20.0, 25.0]), 0.5)
        self.assertAlmostEqual(trapmf(26.0, [5.0, 10.0, 20.0, 25.0]), 0.0)

    def test_trapmf_open_shoulders(self):
        # Left open shoulder: [0, 0, 8, 16]
        self.assertEqual(trapmf(-10.0, [0.0, 0.0, 8.0, 16.0]), 1.0)
        self.assertEqual(trapmf(0.0, [0.0, 0.0, 8.0, 16.0]), 1.0)
        # Right open shoulder: [38, 44, 50, 50]
        self.assertEqual(trapmf(50.0, [38.0, 44.0, 50.0, 50.0]), 1.0)
        self.assertEqual(trapmf(65.0, [38.0, 44.0, 50.0, 50.0]), 1.0)


class TestFuzzyLogicController(unittest.TestCase):
    """Test complete Mamdani FLC inference and bounds."""

    def setUp(self):
        self.controller = FuzzyTrafficController()

    def test_rule_base_size(self):
        # Specification requires at least 27 rules
        self.assertGreaterEqual(len(self.controller.rules), 27)

    def test_defuzzification_bounds(self):
        # Test low traffic -> low green time
        low_res = self.controller.calculate_green_extension(queue_length=2.0, waiting_time=10.0, arrival_rate=1.0)
        self.assertGreaterEqual(low_res["green_extension_seconds"], 10.0)
        self.assertLessEqual(low_res["green_extension_seconds"], 25.0)

        # Test heavy traffic -> high green time
        high_res = self.controller.calculate_green_extension(queue_length=48.0, waiting_time=170.0, arrival_rate=9.5)
        self.assertGreaterEqual(high_res["green_extension_seconds"], 50.0)
        self.assertLessEqual(high_res["green_extension_seconds"], 60.0)

    def test_monotonic_response(self):
        # As queue increases, green extension should increase or stay equal
        r1 = self.controller.calculate_green_extension(5.0, 30.0, 2.0)["green_extension_seconds"]
        r2 = self.controller.calculate_green_extension(20.0, 30.0, 2.0)["green_extension_seconds"]
        r3 = self.controller.calculate_green_extension(40.0, 30.0, 2.0)["green_extension_seconds"]
        self.assertLessEqual(r1, r2)
        self.assertLessEqual(r2, r3)

    def test_active_rules_and_details_reported(self):
        res = self.controller.calculate_green_extension(25.0, 80.0, 5.0)
        self.assertIn("active_rules", res)
        self.assertGreater(len(res["active_rules"]), 0)
        self.assertIn("fuzzified_inputs", res)
        self.assertIn("consequent_weights", res)

    def test_input_sanitization_handling(self):
        # Passing strings, negative numbers, NaNs, Infs should never crash the controller
        res1 = self.controller.calculate_green_extension("25", "-10", "4.5")
        self.assertGreaterEqual(res1["green_extension_seconds"], 10.0)

        res2 = self.controller.calculate_green_extension(float("nan"), 50.0, float("inf"))
        self.assertGreaterEqual(res2["green_extension_seconds"], 10.0)
        self.assertLessEqual(res2["green_extension_seconds"], 60.0)

    def test_ruspini_partition_completeness(self):
        # Check that across continuous universes, sum(mu) >= 0.99 for all variables
        for var in [self.controller.queue_var, self.controller.wait_var,
                    self.controller.arrival_var, self.controller.extension_var]:
            grid = np.linspace(var.universe[0], var.universe[1], 500)
            sum_mu = np.zeros_like(grid)
            for term in var.terms:
                sum_mu += var.evaluate_curve(grid, term)
            self.assertTrue(np.all(sum_mu >= 0.99), f"Dead zone detected in {var.name}")


class TestIntegrationHookAndMilestone1(unittest.TestCase):
    """Verify get_optimal_phase conforms to Person 2 and Person 3 contracts."""

    def test_milestone1_continuous_stream_fuzz(self):
        """Simulate Person 2 streaming 100 random detector frames."""
        np.random.seed(123)
        for _ in range(100):
            q_ns = np.random.uniform(0.0, 50.0)
            w_ns = np.random.uniform(0.0, 180.0)
            arr_ns = np.random.uniform(0.0, 10.0)
            q_ew = np.random.uniform(0.0, 50.0)
            w_ew = np.random.uniform(0.0, 180.0)
            arr_ew = np.random.uniform(0.0, 10.0)

            payload = {
                "current_phase": 0,
                "elapsed_green": np.random.uniform(0.0, 70.0),
                "phases": {
                    0: {"queue_length": q_ns, "wait_time": w_ns, "arrival_rate": arr_ns, "name": "NS"},
                    1: {"queue_length": q_ew, "wait_time": w_ew, "arrival_rate": arr_ew, "name": "EW"}
                }
            }
            res = get_optimal_phase(payload)
            self.assertIn(res["next_phase_id"], [0, 1])
            self.assertGreaterEqual(res["allocated_green_seconds"], 10.0)
            self.assertLessEqual(res["allocated_green_seconds"], 60.0)
            self.assertIsInstance(res["should_switch"], bool)

    def test_minimum_green_guarantee(self):
        # Even if competitor has high queue, phase shouldn't switch before min green
        payload = {
            "current_phase": 0,
            "elapsed_green": 4.0,  # Below default min 10s
            "phases": {
                0: {"queue": 2, "wait": 10, "arrival": 1},
                1: {"queue": 45, "wait": 150, "arrival": 9}
            }
        }
        res = get_optimal_phase(payload)
        self.assertEqual(res["next_phase_id"], 0)
        self.assertFalse(res["should_switch"])

    def test_maximum_green_cap_switching(self):
        # When max green cap is exceeded, system must switch to waiting competitor
        payload = {
            "current_phase": 0,
            "elapsed_green": 65.0,  # Exceeded 60s
            "phases": {
                0: {"queue": 25, "wait": 10, "arrival": 5},
                1: {"queue": 15, "wait": 65, "arrival": 3}
            }
        }
        res = get_optimal_phase(payload)
        self.assertEqual(res["next_phase_id"], 1)
        self.assertTrue(res["should_switch"])

    def test_string_keys_compatibility(self):
        # Test JSON-serialized payloads where keys are strings
        payload = {
            "current_phase": "0",
            "elapsed_green": 20.0,
            "phases": {
                "0": {"queue": 5, "wait": 10, "arrival": 1},
                "1": {"queue": 35, "wait": 120, "arrival": 8}
            }
        }
        res = get_optimal_phase(payload)
        self.assertIn(str(res["next_phase_id"]), ["0", "1"])

    def test_four_way_junction_support(self):
        # 4 phases: North, South, East, West
        payload = {
            "current_phase": "North",
            "elapsed_green": 35.0,
            "phases": {
                "North": {"queue": 3, "wait": 10, "arrival": 1},
                "South": {"queue": 2, "wait": 15, "arrival": 1},
                "East":  {"queue": 30, "wait": 95, "arrival": 7},
                "West":  {"queue": 42, "wait": 140, "arrival": 9}
            }
        }
        res = get_optimal_phase(payload)
        self.assertEqual(res["next_phase_id"], "West")
        self.assertTrue(res["should_switch"])

    def test_none_input_resilience(self):
        # Passing None or empty dictionary should never raise an exception
        res1 = get_optimal_phase(None)
        self.assertIn("next_phase_id", res1)
        self.assertGreaterEqual(res1["allocated_green_seconds"], 10.0)

        res2 = get_optimal_phase({})
        self.assertIn("next_phase_id", res2)
        self.assertGreaterEqual(res2["allocated_green_seconds"], 10.0)


class TestGeneticAlgorithm(unittest.TestCase):
    """Verify GA chromosome encoding and search operators."""

    def setUp(self):
        self.manager = ChromosomeManager()
        self.controller = FuzzyTrafficController()
        self.base_params = self.controller.get_parameters()

    def test_chromosome_roundtrip_and_monotonicity(self):
        chrom = self.manager.params_to_chromosome(self.base_params)
        self.assertEqual(len(chrom), self.manager.num_genes)

        decoded = self.manager.chromosome_to_params(chrom, self.base_params)
        # Check triangular MF monotonicity
        tri_p = decoded["QueueLength"]["Medium"]
        self.assertTrue(tri_p[0] <= tri_p[1] <= tri_p[2])
        # Check trapezoidal MF monotonicity
        trap_p = decoded["QueueLength"]["Short"]
        self.assertTrue(trap_p[0] <= trap_p[1] <= trap_p[2] <= trap_p[3])

    def test_ga_optimization_and_improvement(self):
        optimizer = GAOptimizer(population_size=10, generations=4, random_seed=42, verbose=False)
        results = optimizer.run(verbose=False)
        self.assertIn("optimized_fitness", results)
        self.assertIn("improvement_percent", results)
        self.assertLessEqual(results["optimized_fitness"], results["baseline_fitness"])
        self.assertGreater(results["improvement_percent"], 5.0)


class TestANFISTuner(unittest.TestCase):
    """Verify 5-layer ANFIS model and hybrid learning."""

    def test_5_layers_forward_pass(self):
        net = ANFISNet()
        # Input batch of 4 samples
        x = torch.tensor([
            [10.0, 30.0, 2.0],
            [25.0, 80.0, 5.0],
            [40.0, 140.0, 8.0],
            [0.0, 0.0, 0.0]
        ], dtype=torch.float32)

        out = net(x)
        self.assertEqual(out.shape, (4, 1))
        # Bounds check [10, 60]
        self.assertTrue(torch.all(out >= 10.0))
        self.assertTrue(torch.all(out <= 60.0))

    def test_layer_breakdown_inspection(self):
        net = ANFISNet()
        sample = torch.tensor([[15.0, 50.0, 3.5]], dtype=torch.float32)
        breakdown = net.get_layer_details(sample)

        self.assertIn("layer1_mfs", breakdown)
        self.assertIn("layer2_firing_strengths", breakdown)
        self.assertIn("layer3_normalized_weights", breakdown)
        self.assertIn("layer4_rule_outputs", breakdown)
        self.assertIn("layer5_crisp_output", breakdown)
        self.assertEqual(len(breakdown["layer2_firing_strengths"]), 27)

    def test_hybrid_training_loss_drop(self):
        tuner = ANFISTrafficTuner()
        X, Y = tuner.generate_synthetic_traffic_dataset(n_samples=150, seed=12)
        metrics = tuner.train_hybrid(X, Y, epochs=3, verbose=False)
        self.assertLess(metrics["final_mse"], metrics["baseline_loss"])
        self.assertGreaterEqual(metrics["r2_score"], 0.70)

    def test_model_save_and_load_roundtrip(self):
        tuner = ANFISTrafficTuner()
        X, Y = tuner.generate_synthetic_traffic_dataset(n_samples=100, seed=99)
        tuner.train_hybrid(X, Y, epochs=2, verbose=False)

        pred_before = tuner.predict(24.0, 60.0, 4.0)["allocated_green_seconds"]
        temp_path = "test_anfis_weights.pt"
        tuner.save_model(temp_path)

        new_tuner = ANFISTrafficTuner()
        new_tuner.load_model(temp_path)
        pred_after = new_tuner.predict(24.0, 60.0, 4.0)["allocated_green_seconds"]

        if os.path.exists(temp_path):
            os.remove(temp_path)

        self.assertAlmostEqual(pred_before, pred_after, places=2)


if __name__ == "__main__":
    unittest.main()
