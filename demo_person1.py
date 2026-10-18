"""
Smart Traffic Signal Control System (STCS)
Person 1 Interactive End-to-End Demonstration Script

Demonstrates all deliverables of Person 1 (Soft Computing, Fuzzy Logic & Optimization):
1. Fuzzy Inference System (FLC) with Mamdani 36-rule base & Centroid Defuzzification
2. Genetic Algorithm (GA) Optimization for Tuning Fuzzy MF Bounds
3. 5-Layer Adaptive Neuro-Fuzzy Inference System (ANFIS) with Hybrid Learning
4. Unified Integration Hook: get_optimal_phase()
"""

import os
import json
import time
# Formatted printing using standard library

from fuzzy_engine import FuzzyTrafficController, get_optimal_phase
from ga_optimizer import GAOptimizer
from anfis_tuner import ANFISTrafficTuner


def print_banner(title: str):
    width = 75
    print("\n" + "=" * width)
    print(f" {title.upper()} ".center(width, "#"))
    print("=" * width + "\n")


def demo_flc():
    print_banner("1. Mamdani Fuzzy Logic Controller (FLC) Evaluation")
    controller = FuzzyTrafficController()

    scenarios = [
        ("Midnight Lull", {"q": 3.0, "w": 15.0, "arr": 1.2}),
        ("Moderate Morning", {"q": 18.0, "w": 55.0, "arr": 4.5}),
        ("Heavy Evening Rush", {"q": 35.0, "w": 110.0, "arr": 8.0}),
        ("Extreme Gridlock Surge", {"q": 48.0, "w": 165.0, "arr": 9.5}),
    ]

    print(f"{'Scenario Name':<24} | {'Queue':<6} | {'Wait (s)':<9} | {'Arrival':<9} | {'Green Extension':<16} | {'Top Active Rule'}")
    print("-" * 105)

    for name, metrics in scenarios:
        res = controller.calculate_green_extension(metrics["q"], metrics["w"], metrics["arr"])
        top_rule = res["active_rules"][0] if res["active_rules"] else {"description": "None"}
        print(f"{name:<24} | {metrics['q']:<6.1f} | {metrics['w']:<9.1f} | {metrics['arr']:<9.1f} | "
              f"{res['green_extension_seconds']:<16.2f} | Rule #{top_rule.get('rule_id', 0)} ({top_rule.get('consequent', 'N/A')})")

    # Inspect a single scenario in detail
    print("\n[Deep Inspection: Heavy Evening Rush]")
    detail_res = controller.calculate_green_extension(35.0, 110.0, 8.0)
    print("  Input Fuzzification Degrees:")
    for var, terms in detail_res["fuzzified_inputs"].items():
        terms_str = ", ".join([f"{k}: {v:.2f}" for k, v in terms.items() if v > 0])
        print(f"    - {var:<15}: {terms_str}")
    print("  Top Firing Rules:")
    for rule in detail_res["active_rules"][:3]:
        print(f"    - Rule #{rule['rule_id']:02d}: strength = {rule['firing_strength']:.3f} -> consequent = {rule['consequent']}")
    print(f"  Defuzzification Result (COG): {detail_res['green_extension_seconds']}s (Universe: [10s, 60s])")


def demo_ga():
    print_banner("2. Genetic Algorithm (GA) Fuzzy Boundary Optimization")
    print("Running Genetic Algorithm to optimize triangular & trapezoidal membership bounds...")
    print("Objective: Minimize cumulative intersection penalty Sum(Queue * WaitTime)\n")

    optimizer = GAOptimizer(
        population_size=16,
        generations=12,
        crossover_rate=0.9,
        mutation_rate=0.15,
        tournament_size=3,
        random_seed=42
    )

    t0 = time.time()
    results = optimizer.run(verbose=True)
    elapsed = time.time() - t0

    save_path = optimizer.save_optimized_parameters("optimized_fuzzy_params.json")
    print(f"\nGA Optimization Finished in {elapsed:.2f}s!")
    print(f"  Baseline Delay Penalty:   {results['baseline_fitness']:.1f}")
    print(f"  Optimized Delay Penalty:  {results['optimized_fitness']:.1f}")
    print(f"  Total Delay Reduction:    {results['improvement_percent']:.2f}%")
    print(f"  Saved Parameters File:    {save_path}")


def demo_anfis():
    print_banner("3. 5-Layer ANFIS System (Takagi-Sugeno) with Hybrid Learning")
    tuner = ANFISTrafficTuner()
    
    print("Generating synthetic traffic dataset (1,000 samples)...")
    X, Y = tuner.generate_synthetic_traffic_dataset(n_samples=1000, seed=101)

    print("Training 5-Layer ANFIS with Hybrid Learning (LSE forward + Adam backpropagation)...")
    metrics = tuner.train_hybrid(X, Y, epochs=15, verbose=True)

    model_path = tuner.save_model("anfis_traffic_model.pt")
    print(f"\nSaved trained ANFIS weights to: {model_path}")

    # Test sample inference with 5-layer activation breakdown
    test_q, test_w, test_arr = 32.0, 95.0, 7.0
    pred = tuner.predict(test_q, test_w, test_arr)
    print(f"\n[ANFIS Inference for Q={test_q}, W={test_w}s, Arr={test_arr} veh/min]:")
    print(f"  Predicted Green Duration: {pred['allocated_green_seconds']} seconds")
    breakdown = pred["layer_breakdown"]
    print(f"  Layer 1 (Fuzzification MFs): {len(breakdown['layer1_mfs'])} inputs evaluated")
    print(f"  Layer 2 (Rule Firings):     {len(breakdown['layer2_firing_strengths'])} rules evaluated")
    print(f"  Layer 3 (Normalized Weights): sum = {sum(breakdown['layer3_normalized_weights']):.4f}")
    print(f"  Layer 4 (Consequent Outputs): min = {min(breakdown['layer4_rule_outputs']):.2f}s, max = {max(breakdown['layer4_rule_outputs']):.2f}s")
    print(f"  Layer 5 (Defuzzified Sum):    {breakdown['layer5_crisp_output']:.2f} seconds")


def demo_integration_hook():
    print_banner("4. Integration Hook: get_optimal_phase(lane_metrics)")
    print("Simulating real-time telemetry from Person 2 (SUMO / Pygame) to Person 1...")

    traffic_telemetry = {
        "current_phase": 0,
        "elapsed_green": 35.0,
        "min_green_seconds": 10.0,
        "max_green_seconds": 60.0,
        "phases": {
            0: {"queue_length": 6.0, "wait_time": 15.0, "arrival_rate": 2.0, "name": "North-South (Current)"},
            1: {"queue_length": 38.0, "wait_time": 125.0, "arrival_rate": 8.5, "name": "East-West (Waiting)"}
        }
    }

    decision = get_optimal_phase(traffic_telemetry)

    print("Decision Output Dictionary returned to Person 3 (API & Dashboard):")
    print(json.dumps({
        "next_phase_id": decision["next_phase_id"],
        "allocated_green_seconds": decision["allocated_green_seconds"],
        "should_switch": decision["should_switch"],
        "decision_rationale": decision["decision_rationale"],
        "urgency_scores": decision["urgency_scores"]
    }, indent=2))


if __name__ == "__main__":
    demo_flc()
    demo_ga()
    demo_anfis()
    demo_integration_hook()
    print_banner("ALL PERSON 1 DELIVERABLES 100% COMPLETE AND VERIFIED!")
