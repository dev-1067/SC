"""
Automated Verification & Unit Test Suite for Person 3: Full-Stack Infrastructure, Orchestration & Dashboard
Exhaustive verification of Person 3 deliverables:
1. Database & Telemetry Storage (database.py):
   - SQLite Database & SQLAlchemy ORM mapping
   - MetricLog telemetry persistence and chronological query retrieval
   - SignalSwitch audit trail logging with Soft Computing decision rationales
   - BenchmarkRecord persistence and compliance targets evaluation
2. FastAPI Backend & REST API (main.py):
   - /api/status: Telemetry snapshot, signal states, fuzzy inspector, ANN forecast
   - /api/control/mode: Mode switching (ADAPTIVE, STATIC_FIXED, MANUAL)
   - /api/control/switch: Manual phase arbitration with yellow clearance safety
   - /api/control/emergency: Emergency vehicle priority preemption & corridor clearance
   - /api/control/demand: Dynamic inflow arrival rate adjustment
   - /api/metrics/history & /api/switches/history: Audit trail endpoints
   - /api/soft-computing/models: Model architecture metadata
   - Static dashboard route /: HTML5 web visualizer delivery
3. Real-Time WebSockets (/ws/traffic-stream):
   - Live telemetry packet broadcast contract validation
4. Live Comparative Benchmark Integration:
   - Automated 30-second benchmark execution and scorecard verification
"""

import json
import os
import unittest
from fastapi.testclient import TestClient

from database import DatabaseManager, get_db
from orchestrator import SimulationOrchestrator, get_orchestrator
from main import app


class TestDatabaseStorage(unittest.TestCase):
    """Test SQLite database operations and SQLAlchemy ORM models."""

    def setUp(self):
        self.db = get_db()
        self.db.clear_metrics()

    def test_metric_logging_and_retrieval(self):
        state = {
            "sim_time": 10.5,
            "control_mode": "ADAPTIVE",
            "active_phase": 0,
            "phase_name": "North-South",
            "signal_states": {"NS": "GREEN", "EW": "RED"},
            "lanes": {
                "North": {"queue_length": 3, "wait_time": 12.0},
                "South": {"queue_length": 2, "wait_time": 8.0},
                "East": {"queue_length": 5, "wait_time": 25.0},
                "West": {"queue_length": 4, "wait_time": 22.0}
            },
            "phases": {
                0: {"queue_length": 5, "wait_time": 12.0, "name": "North-South"},
                1: {"queue_length": 9, "wait_time": 25.0, "name": "East-West"}
            },
            "total_queued": 14,
            "total_throughput": 6,
            "average_delay_seconds": 7.4,
            "active_vehicle_count": 18,
            "ann_forecast": {
                "0": {"predicted_queue_10min": 15.2, "surge_factor": 0.35},
                "1": {"predicted_queue_10min": 24.8, "surge_factor": 0.62}
            }
        }
        log_id = self.db.log_metric(state)
        self.assertIsNotNone(log_id)
        self.assertGreater(log_id, 0)

        records = self.db.get_recent_metrics(limit=10)
        self.assertGreaterEqual(len(records), 1)
        latest = records[-1]
        self.assertEqual(latest["sim_time"], 10.5)
        self.assertEqual(latest["control_mode"], "ADAPTIVE")
        self.assertEqual(latest["total_queued"], 14)
        self.assertEqual(latest["total_throughput"], 6)

    def test_signal_switch_audit_logging(self):
        switch_id = self.db.log_signal_switch(
            sim_time=45.0,
            control_mode="ADAPTIVE",
            from_phase=0,
            to_phase=1,
            from_name="North-South",
            to_name="East-West",
            allocated_green=42.5,
            trigger_reason="FLC_URGENCY_ARBITRATION",
            decision_rationale="Competitor urgency (58.2) exceeded current phase (14.0).",
            active_rules=[{"rule_id": 24, "firing_strength": 0.85, "consequent": "VeryLong"}]
        )
        self.assertIsNotNone(switch_id)

        switches = self.db.get_recent_switches(limit=5)
        self.assertGreaterEqual(len(switches), 1)
        sw = switches[0]
        self.assertEqual(sw["from_name"], "North-South")
        self.assertEqual(sw["to_name"], "East-West")
        self.assertEqual(sw["allocated_green"], 42.5)
        self.assertIn("Competitor urgency", sw["decision_rationale"])
        self.assertGreater(len(sw["active_rules"]), 0)

    def test_benchmark_persistence(self):
        bench_data = {
            "benchmark_name": "Comparative Surge Benchmark",
            "duration_seconds": 150.0,
            "static_avg_delay": 7.1,
            "adaptive_avg_delay": 5.0,
            "delay_reduction_pct": 29.58,
            "static_throughput": 4,
            "adaptive_throughput": 7,
            "static_queued": 16,
            "adaptive_queued": 6,
            "notes": "Verified Soft Computing compliance across Modules 1 to 6."
        }
        b_id = self.db.save_benchmark(bench_data)
        self.assertIsNotNone(b_id)

        history = self.db.get_benchmarks(limit=5)
        self.assertGreaterEqual(len(history), 1)
        latest_b = history[0]
        self.assertEqual(latest_b["delay_reduction_pct"], 29.58)
        self.assertTrue(latest_b["target_met"])


class TestFastAPIRoutes(unittest.TestCase):
    """Test FastAPI REST endpoints and status responses."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_status_endpoint(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        
        # Telemetry contract verification
        required_fields = [
            "sim_time", "control_mode", "active_phase", "phase_name",
            "signal_states", "elapsed_in_phase", "allocated_green",
            "phases", "lanes", "total_queued", "total_throughput",
            "average_delay_seconds", "ann_forecast", "fuzzy_inspector"
        ]
        for f in required_fields:
            self.assertIn(f, data, f"Missing telemetry field: {f}")

        # Fuzzy inspector structure verification
        fi = data["fuzzy_inspector"]
        self.assertIn("fuzzified_inputs", fi)
        self.assertIn("active_rules", fi)
        self.assertIn("green_extension_seconds", fi)

    def test_control_mode_toggle(self):
        # Switch to STATIC_FIXED
        res1 = self.client.post("/api/control/mode", json={"mode": "STATIC_FIXED"})
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.json()["control_mode"], "STATIC_FIXED")

        # Verify status reflected
        status = self.client.get("/api/status").json()
        self.assertEqual(status["control_mode"], "STATIC_FIXED")

        # Switch back to ADAPTIVE
        res2 = self.client.post("/api/control/mode", json={"mode": "ADAPTIVE"})
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json()["control_mode"], "ADAPTIVE")

    def test_invalid_mode_handling(self):
        res = self.client.post("/api/control/mode", json={"mode": "NON_EXISTENT_MODE"})
        self.assertEqual(res.status_code, 400)

    def test_manual_phase_switch(self):
        res = self.client.post("/api/control/switch", json={"target_phase": 1})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "SUCCESS")

    def test_emergency_preemption(self):
        # Trigger emergency on West corridor
        res = self.client.post("/api/control/emergency", json={"approach": "West", "duration": 15.0})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["emergency_active"])
        self.assertEqual(res.json()["approach"], "West")

        # Check status shows emergency active
        status = self.client.get("/api/status").json()
        self.assertTrue(status["emergency_active"])
        self.assertEqual(status["emergency_approach"], "West")

        # Clear emergency
        clear_res = self.client.post("/api/control/emergency/clear")
        self.assertEqual(clear_res.status_code, 200)
        self.assertFalse(clear_res.json()["emergency_active"])

    def test_demand_adjustment(self):
        res = self.client.post("/api/control/demand", json={
            "north": 9.0, "south": 8.5, "east": 3.0, "west": 2.5
        })
        self.assertEqual(res.status_code, 200)
        demands = res.json()["arrival_rates"]
        self.assertEqual(demands["North"], 9.0)
        self.assertEqual(demands["South"], 8.5)
        self.assertEqual(demands["East"], 3.0)
        self.assertEqual(demands["West"], 2.5)

    def test_simulation_speed_and_pause_resume(self):
        # Speed adjustment
        res_spd = self.client.post("/api/control/speed", json={"speed": 2.0})
        self.assertEqual(res_spd.status_code, 200)
        self.assertEqual(res_spd.json()["sim_speed"], 2.0)

        # Pause
        res_p = self.client.post("/api/control/pause")
        self.assertEqual(res_p.status_code, 200)
        self.assertTrue(res_p.json()["is_paused"])

        # Resume
        res_r = self.client.post("/api/control/resume")
        self.assertEqual(res_r.status_code, 200)
        self.assertFalse(res_r.json()["is_paused"])

    def test_soft_computing_models_introspection(self):
        res = self.client.get("/api/soft-computing/models")
        self.assertEqual(res.status_code, 200)
        info = res.json()
        self.assertIn("fuzzy_controller", info)
        self.assertIn("ann_predictor", info)
        self.assertGreaterEqual(info["fuzzy_controller"]["rule_count"], 27)

    def test_dashboard_static_route(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("STCS", res.text)
        self.assertIn("trafficCanvas", res.text)


class TestWebSocketTelemetry(unittest.TestCase):
    """Test WebSocket high-frequency stream connection and payload."""

    def test_websocket_stream_initial_handshake(self):
        client = TestClient(app)
        with client.websocket_connect("/ws/traffic-stream") as websocket:
            data = websocket.receive_text()
            payload = json.loads(data)
            self.assertIn("sim_time", payload)
            self.assertIn("control_mode", payload)
            self.assertIn("active_phase", payload)
            self.assertIn("signal_states", payload)
            self.assertIn("phases", payload)
            self.assertIn("fuzzy_inspector", payload)


class TestBenchmarkExecution(unittest.TestCase):
    """Test Comparative Benchmark Runner (Person 1 + 2 + 3 Integration)."""

    def test_run_benchmark_endpoint(self):
        client = TestClient(app)
        # Run a 30s benchmark test
        res = client.post("/api/benchmark/run", json={"duration_seconds": 30.0})
        self.assertEqual(res.status_code, 200)
        report = res.json()
        
        self.assertIn("static_avg_delay", report)
        self.assertIn("adaptive_avg_delay", report)
        self.assertIn("delay_reduction_pct", report)
        self.assertIn("static_throughput", report)
        self.assertIn("adaptive_throughput", report)


if __name__ == "__main__":
    unittest.main()
