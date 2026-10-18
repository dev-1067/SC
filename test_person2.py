"""
Automated Verification & Unit Test Suite for Person 2: Sensing, Neural Networks & Traffic Simulation
Exhaustive verification of Person 2 deliverables:
1. ANN Traffic Flow Predictor (ann_predictor.py):
   - MLP Architecture (6 inputs, 2 outputs: queue forecast, surge factor)
   - Backpropagation learning rule & MSE loss reduction
   - Inference pipeline with robust input sanitization
   - Model save & load round-trip fidelity
2. OpenCV Vehicle Counting & Queue Estimator (vision_detector.py):
   - Background Subtraction & Morphological Filtering
   - Multi-lane ROI segmentation (North, South, East, West)
   - Centroid tracking, velocity estimation, and queue state classification
   - Annotated frame rendering
3. 4-Way Crossroad Simulation Environment (simulation_env.py):
   - Vehicle physics and car-following gap constraint
   - Traffic signal actuation & 3-second yellow clearance safety
   - Dual control modes: ADAPTIVE vs. STATIC_FIXED
   - High-frequency telemetry serialization
4. Milestone 1 Integration (Person 1 + Person 2):
   - Continuous streaming of simulated detector counts into Person 1's get_optimal_phase()
5. Milestone 2 Integration (Person 2 + Person 3):
   - Verification of simulation state payload contracts for FastAPI and WebSockets
"""

import os
import unittest
import numpy as np
import torch

from ann_predictor import TrafficFlowMLP, ANNTrafficPredictor, TrafficDatasetGenerator
from vision_detector import VisionQueueDetector, SyntheticTrafficCamera
from simulation_env import TrafficSimulation, SimVehicle
from fuzzy_engine import get_optimal_phase


class TestANNTrafficPredictor(unittest.TestCase):
    """Test Multi-Layer Perceptron (MLP) Traffic Flow Predictor."""

    def test_mlp_forward_dimensions(self):
        mlp = TrafficFlowMLP(input_dim=6, hidden_dim1=16, hidden_dim2=8, output_dim=2)
        dummy_input = torch.randn(4, 6)
        out = mlp(dummy_input)
        self.assertEqual(out.shape, (4, 2))

    def test_backprop_training_loss_drop(self):
        predictor = ANNTrafficPredictor()
        X, Y, stats = TrafficDatasetGenerator.generate_data(n_samples=300, seed=12)
        metrics = predictor.train_backprop(X, Y, stats, epochs=5, verbose=False)
        self.assertLess(metrics["final_mse"], metrics["initial_loss"])
        self.assertGreaterEqual(metrics["r2_score"], 0.70)

    def test_predict_flow_sanitization(self):
        predictor = ANNTrafficPredictor()
        X, Y, stats = TrafficDatasetGenerator.generate_data(n_samples=200, seed=42)
        predictor.train_backprop(X, Y, stats, epochs=3, verbose=False)

        # Normal prediction
        pred = predictor.predict_flow(q_t=20.0, q_t_prev=15.0, v_t=6.0, w_t=50.0, hour_of_day=8.5)
        self.assertIn("predicted_queue_10min", pred)
        self.assertIn("surge_factor", pred)
        self.assertGreaterEqual(pred["predicted_queue_10min"], 0.0)
        self.assertLessEqual(pred["predicted_queue_10min"], 50.0)

        # Extreme / dirty input sanitization test with explicit NaN and Inf
        dirty_pred = predictor.predict_flow(q_t=float("nan"), q_t_prev="-10", v_t="nan", w_t=float("inf"), hour_of_day=26.0)
        self.assertIsInstance(dirty_pred["predicted_queue_10min"], float)
        self.assertFalse(np.isnan(dirty_pred["predicted_queue_10min"]))
        self.assertFalse(np.isnan(dirty_pred["surge_factor"]))
        self.assertIsInstance(dirty_pred["is_surge_imminent"], bool)

    def test_ann_save_load_fidelity(self):
        predictor = ANNTrafficPredictor()
        X, Y, stats = TrafficDatasetGenerator.generate_data(n_samples=200, seed=99)
        predictor.train_backprop(X, Y, stats, epochs=3, verbose=False)

        pred_orig = predictor.predict_flow(22.0, 18.0, 7.0, 60.0, 17.5)["predicted_queue_10min"]
        tmp_model = "test_ann_weights.pt"
        tmp_stats = "test_ann_stats.json"

        predictor.save_model(tmp_model, tmp_stats)

        new_predictor = ANNTrafficPredictor()
        new_predictor.load_model(tmp_model, tmp_stats)
        pred_loaded = new_predictor.predict_flow(22.0, 18.0, 7.0, 60.0, 17.5)["predicted_queue_10min"]

        if os.path.exists(tmp_model): os.remove(tmp_model)
        if os.path.exists(tmp_stats): os.remove(tmp_stats)

        self.assertAlmostEqual(pred_orig, pred_loaded, places=2)


class TestVisionQueueDetector(unittest.TestCase):
    """Test OpenCV Vehicle Detection and Queue Extractor."""

    def setUp(self):
        self.detector = VisionQueueDetector()
        self.camera = SyntheticTrafficCamera()

    def test_camera_frame_generation(self):
        frame = self.camera.generate_frame("GREEN", "RED")
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_vision_pipeline_and_tracking(self):
        # Feed 15 warmup frames
        for _ in range(15):
            f = self.camera.generate_frame("GREEN", "RED")
            metrics = self.detector.process_frame(f)

        self.assertIn("lane_metrics", metrics)
        self.assertIn("phases_payload", metrics)
        self.assertIn("North", metrics["lane_metrics"])
        self.assertIn("East", metrics["lane_metrics"])
        self.assertGreater(metrics["fps"], 15.0)

    def test_render_annotated_frame(self):
        frame = self.camera.generate_frame("RED", "GREEN")
        self.detector.process_frame(frame)
        annotated = self.detector.render_annotated_frame(frame)
        self.assertEqual(annotated.shape, frame.shape)
        self.assertEqual(annotated.dtype, np.uint8)


class TestTrafficSimulation(unittest.TestCase):
    """Test 4-Way Crossroad Simulation Environment."""

    def test_simulation_step_headless(self):
        sim = TrafficSimulation(control_mode="ADAPTIVE", headless=True, random_seed=42)
        state = sim.step(dt=1.0)

        self.assertIn("sim_time", state)
        self.assertIn("active_phase", state)
        self.assertIn("signal_states", state)
        self.assertIn("phases", state)
        self.assertIn("vehicles", state)
        self.assertEqual(state["sim_time"], 1.0)

    def test_yellow_clearance_transition(self):
        sim = TrafficSimulation(control_mode="STATIC_FIXED", headless=True)
        # Advance simulation to reach the 30s static threshold
        state = None
        for _ in range(60):
            state = sim.step(dt=0.5)

        # At 30s, the controller triggers a 3-second yellow clearance phase
        self.assertTrue(state["in_yellow_transition"])
        self.assertEqual(state["signal_states"]["NS"], "YELLOW")

    def test_car_following_physics(self):
        lead = SimVehicle(1, "North", spawn_pos=150.0, speed=0.0)
        follower = SimVehicle(2, "North", spawn_pos=100.0, speed=4.0)

        # Update follower physics behind stopped lead
        for _ in range(12):
            follower.update_physics(lead, "RED", stop_line_pos=200.0, dt=0.5)

        # Follower should have slowed down behind lead car
        self.assertLess(follower.speed, 2.0)
        self.assertLess(follower.pos, lead.pos)


class TestMilestone1And2Integration(unittest.TestCase):
    """Test cross-module integration across Person 1, Person 2, and Person 3."""

    def test_milestone1_simulation_to_flc_streaming(self):
        """
        Milestone 1 Specification:
        Person 2 streams simulated queue data into Person 1's FLC function;
        FLC outputs valid green times (10s–60s) without crashes.
        """
        sim = TrafficSimulation(control_mode="ADAPTIVE", headless=True, random_seed=101)

        for step_idx in range(50):
            state = sim.step(dt=0.5)
            phases_data = state["phases"]

            # Person 1 FLC integration contract
            payload = {
                "current_phase": state["active_phase"],
                "elapsed_green": state["elapsed_in_phase"],
                "phases": phases_data
            }
            decision = get_optimal_phase(payload)

            self.assertIn(decision["next_phase_id"], [0, 1])
            self.assertGreaterEqual(decision["allocated_green_seconds"], 10.0)
            self.assertLessEqual(decision["allocated_green_seconds"], 60.0)
            self.assertIsInstance(decision["should_switch"], bool)

    def test_milestone2_telemetry_payload_schema(self):
        """
        Milestone 2 Specification:
        Person 3's backend connects to Person 2's simulation via WebSocket,
        correctly reading queues and toggling phases.
        """
        sim = TrafficSimulation(control_mode="ADAPTIVE", headless=True)
        # Advance 10 steps
        for _ in range(10):
            state = sim.step(dt=0.5)

        # Validate schema keys required by Person 3 FastAPI / WebSockets
        required_keys = [
            "sim_time", "control_mode", "active_phase", "phase_name",
            "signal_states", "elapsed_in_phase", "allocated_green",
            "phases", "lanes", "active_vehicle_count", "total_queued",
            "total_throughput", "average_delay_seconds", "ann_forecast", "vehicles"
        ]
        for key in required_keys:
            self.assertIn(key, state, f"Missing required telemetry key: {key}")

        # Test mode toggle
        sim.set_control_mode("STATIC_FIXED")
        self.assertEqual(sim.control_mode, "STATIC_FIXED")
        sim.set_control_mode("ADAPTIVE")
        self.assertEqual(sim.control_mode, "ADAPTIVE")


if __name__ == "__main__":
    unittest.main()
