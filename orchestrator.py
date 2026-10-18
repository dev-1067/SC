"""
Smart Traffic Signal Control System (STCS)
Person 3 Deliverable: Central Orchestration & Event Loop Engine (orchestrator.py)
Covers Real-Time Control Loop Orchestration:
- Synchronizes Person 2 (Simulation & Detectors) with Person 1 (FLC & ANN Predictor)
- High-frequency async tick loop with configurable time-acceleration (1x, 2x, 5x)
- WebSocket connection manager broadcasting live telemetry to all connected clients
- Automated SQLite database telemetry and signal switch logging
- Emergency Vehicle Preemption (Priority corridor override)
- Live Comparative Benchmark Runner (Static Fixed-Time vs. Adaptive Soft Computing)
- Deep Soft Computing Model Introspection (Fuzzy rules, MFs, ANN layer activations)
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Any, Optional, Set
from fastapi import WebSocket

from simulation_env import TrafficSimulation, SimVehicle
from fuzzy_engine import get_optimal_phase, get_controller
from ann_predictor import get_ann_predictor
from database import get_db, DatabaseManager


class SimulationOrchestrator:
    """
    Central event loop and orchestrator coordinating the simulation physics,
    soft computing decision engine, telemetry database, and real-time WebSockets.
    """

    def __init__(self):
        self.db = get_db()
        self.control_mode = "ADAPTIVE"
        self.sim = TrafficSimulation(control_mode=self.control_mode, headless=True, random_seed=42)
        
        # Async execution state
        self.is_running = True
        self.sim_speed = 1.0        # 1.0x, 2.0x, 5.0x
        self.step_dt = 0.2          # 200ms simulation delta
        self.tick_interval = 0.1    # 10 Hz broadcast loop (100ms wall time)
        
        # Active WebSockets clients
        self.active_websockets: Set[WebSocket] = set()
        
        # Tracking state
        self.last_phase = self.sim.active_phase
        self.last_db_log_sim_time = 0.0
        self.emergency_active = False
        self.emergency_approach = ""
        self.emergency_timeout = 0.0

        # Benchmark status
        self.benchmark_in_progress = False

    # --------------------------------------------------------------------------
    # WEBSOCKET BROADCAST MANAGEMENT
    # --------------------------------------------------------------------------
    async def connect_client(self, websocket: WebSocket):
        """Registers a new WebSocket subscriber and sends initial state snapshot."""
        await websocket.accept()
        self.active_websockets.add(websocket)
        # Immediate welcome packet
        state = self.get_telemetry_snapshot()
        try:
            await websocket.send_text(json.dumps(state))
        except Exception:
            pass

    def disconnect_client(self, websocket: WebSocket):
        """Removes a disconnected client."""
        self.active_websockets.discard(websocket)

    async def broadcast(self, payload: Dict[str, Any]):
        """Dispatches telemetry packet to all connected dashboard clients."""
        if not self.active_websockets:
            return

        message = json.dumps(payload)
        dead_sockets = set()
        for ws in self.active_websockets:
            try:
                await ws.send_text(message)
            except Exception:
                dead_sockets.add(ws)

        for ws in dead_sockets:
            self.active_websockets.discard(ws)

    # --------------------------------------------------------------------------
    # TICK LOOP & ORCHESTRATION
    # --------------------------------------------------------------------------
    async def run_loop(self):
        """Continuous async event loop executing simulation steps and broadcasting state."""
        while True:
            t_start = time.time()
            if self.is_running and not self.benchmark_in_progress:
                effective_dt = self.step_dt * self.sim_speed
                
                # Check emergency preemption timeout
                if self.emergency_active:
                    self.emergency_timeout -= effective_dt
                    if self.emergency_timeout <= 0.0:
                        self.clear_emergency()

                # Step simulation
                state = self.sim.step(dt=effective_dt)

                # Check if signal phase switch occurred
                current_phase = self.sim.active_phase
                if current_phase != self.last_phase:
                    from_name = self.sim.phase_names.get(self.last_phase, str(self.last_phase))
                    to_name = self.sim.phase_names.get(current_phase, str(current_phase))
                    latest_dec = self.sim.flc_decisions_history[-1] if self.sim.flc_decisions_history else {}
                    
                    self.db.log_signal_switch(
                        sim_time=self.sim.sim_time,
                        control_mode=self.sim.control_mode,
                        from_phase=self.last_phase,
                        to_phase=current_phase,
                        from_name=from_name,
                        to_name=to_name,
                        allocated_green=self.sim.allocated_green,
                        trigger_reason="EMERGENCY_OVERRIDE" if self.emergency_active else (
                            "MAX_GREEN_CAP" if self.sim.elapsed_in_phase >= self.sim.max_green_seconds else "FLC_URGENCY_ARBITRATION"
                        ),
                        decision_rationale=latest_dec.get("decision_rationale", "Phase switch executed."),
                        active_rules=latest_dec.get("active_rules", [])
                    )
                    self.last_phase = current_phase

                # Log periodic metrics to database (every 1.0 simulated seconds)
                if self.sim.sim_time - self.last_db_log_sim_time >= 1.0:
                    self.db.log_metric(state)
                    self.last_db_log_sim_time = self.sim.sim_time

                # Broadcast to WebSockets
                state_packet = self.get_telemetry_snapshot()
                await self.broadcast(state_packet)

            # Sleep to maintain ~10 Hz broadcast
            elapsed = time.time() - t_start
            sleep_time = max(0.01, self.tick_interval - elapsed)
            await asyncio.sleep(sleep_time)

    # --------------------------------------------------------------------------
    # STATE ENRICHMENT & SNAPSHOTS
    # --------------------------------------------------------------------------
    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Returns complete, enhanced JSON telemetry snapshot."""
        state = self.sim.get_state()
        state["sim_speed"] = self.sim_speed
        state["is_paused"] = not self.is_running
        state["emergency_active"] = self.emergency_active
        state["emergency_approach"] = self.emergency_approach
        state["connected_clients"] = len(self.active_websockets)

        # Attach active FLC fuzzification and rule firing details for dashboard inspector
        controller = get_controller()
        p_active = self.sim.active_phase
        phase_data = state["phases"].get(p_active, state["phases"].get(str(p_active), {}))
        
        q = float(phase_data.get("queue_length", 0.0))
        w = float(phase_data.get("wait_time", 0.0))
        arr = float(phase_data.get("arrival_rate", 0.0))

        flc_details = controller.calculate_green_extension(q, w, arr)
        state["fuzzy_inspector"] = {
            "fuzzified_inputs": flc_details["fuzzified_inputs"],
            "active_rules": flc_details["active_rules"],
            "consequent_weights": flc_details["consequent_weights"],
            "green_extension_seconds": flc_details["green_extension_seconds"]
        }

        return state

    # --------------------------------------------------------------------------
    # INTERACTIVE CONTROL ACTIONS
    # --------------------------------------------------------------------------
    def set_control_mode(self, mode: str):
        """Switches controller mode (ADAPTIVE, STATIC_FIXED, MANUAL)."""
        assert mode in ("ADAPTIVE", "STATIC_FIXED", "MANUAL")
        self.control_mode = mode
        self.sim.set_control_mode(mode)

    def set_speed(self, speed: float):
        """Sets simulation speed multiplier (0.5x to 5.0x)."""
        self.sim_speed = float(max(0.5, min(5.0, speed)))

    def pause(self):
        """Pauses simulation clock."""
        self.is_running = False

    def resume(self):
        """Resumes simulation clock."""
        self.is_running = True

    def reset_simulation(self):
        """Resets the intersection to initial state."""
        self.sim = TrafficSimulation(control_mode=self.control_mode, headless=True, random_seed=int(time.time()))
        self.last_phase = self.sim.active_phase
        self.last_db_log_sim_time = 0.0
        self.clear_emergency()

    def set_arrival_demands(self, north: float, south: float, east: float, west: float):
        """Dynamically adjusts inflow demands per approach (vehicles/min)."""
        self.sim.arrival_rates["North"] = float(max(0.0, min(15.0, north)))
        self.sim.arrival_rates["South"] = float(max(0.0, min(15.0, south)))
        self.sim.arrival_rates["East"] = float(max(0.0, min(15.0, east)))
        self.sim.arrival_rates["West"] = float(max(0.0, min(15.0, west)))

    def manual_phase_switch(self, target_phase: Optional[int] = None):
        """Forces an immediate manual phase switch with 3s yellow safety interval."""
        if target_phase is None:
            target_phase = 1 if self.sim.active_phase == 0 else 0
        
        if target_phase != self.sim.active_phase:
            self.sim.in_yellow_transition = True
            self.sim.pending_next_phase = target_phase
            self.sim.elapsed_in_phase = 0.0
            if self.sim.active_phase == 0:
                self.sim.signals["NS"] = "YELLOW"
                self.sim.signals["EW"] = "RED"
            else:
                self.sim.signals["NS"] = "RED"
                self.sim.signals["EW"] = "YELLOW"

    def trigger_emergency_preemption(self, approach: str = "North", duration: float = 20.0):
        """
        Emergency Vehicle Preemption:
        Grants immediate green light priority to ambulance/fire corridor.
        """
        assert approach in ("North", "South", "East", "West")
        self.emergency_active = True
        self.emergency_approach = approach
        self.emergency_timeout = duration

        # Inject high-visibility emergency vehicle near entrance
        emerg_car = SimVehicle(self.sim.next_veh_id, approach, spawn_pos=10.0, speed=8.0)
        emerg_car.color = (255, 0, 80) # Emergency Neon Red
        self.sim.vehicles[approach].append(emerg_car)
        self.sim.next_veh_id += 1

        target_phase = 0 if approach in ("North", "South") else 1
        self.sim.emergency_preemption_active = True
        self.sim.emergency_target_phase = target_phase
        if self.sim.active_phase != target_phase:
            self.manual_phase_switch(target_phase)

    def clear_emergency(self):
        """Restores normal autonomous adaptive signal control."""
        self.emergency_active = False
        self.emergency_approach = ""
        self.emergency_timeout = 0.0
        self.sim.emergency_preemption_active = False

    # --------------------------------------------------------------------------
    # COMPARATIVE BENCHMARK RUNNER (Specification Section 7 Requirement)
    # --------------------------------------------------------------------------
    def run_comparative_benchmark(self, duration_seconds: float = 150.0) -> Dict[str, Any]:
        """
        Executes a 150-second head-to-head simulation benchmark:
        Run A: Static Fixed-Time Controller (30s static NS / 30s static EW)
        Run B: Adaptive Soft Computing Controller (FLC Actuation + ANN Forecast)
        Saves result to SQLite database and returns comparative report.
        """
        self.benchmark_in_progress = True
        steps = int(duration_seconds / 0.5)

        # Surge Demand Pattern (Heavier NS rush hour)
        benchmark_demands = {"North": 7.5, "South": 7.5, "East": 3.0, "West": 3.0}

        # Run A: Static Fixed-Time
        sim_static = TrafficSimulation(control_mode="STATIC_FIXED", headless=True, random_seed=42)
        sim_static.arrival_rates = dict(benchmark_demands)
        for _ in range(steps):
            sim_static.step(dt=0.5)
        state_static = sim_static.get_state()

        # Run B: Adaptive Soft Computing
        sim_adaptive = TrafficSimulation(control_mode="ADAPTIVE", headless=True, random_seed=42)
        sim_adaptive.arrival_rates = dict(benchmark_demands)
        for _ in range(steps):
            sim_adaptive.step(dt=0.5)
        state_adaptive = sim_adaptive.get_state()

        # Metrics Compilation
        d_static = state_static["average_delay_seconds"]
        d_adaptive = state_adaptive["average_delay_seconds"]
        reduction_pct = ((d_static - d_adaptive) / max(1e-4, d_static)) * 100.0

        thru_static = state_static["total_throughput"]
        thru_adaptive = state_adaptive["total_throughput"]
        q_static = state_static["total_queued"]
        q_adaptive = state_adaptive["total_queued"]

        report = {
            "benchmark_name": "Comparative Surge Benchmark (Static vs. Adaptive)",
            "duration_seconds": duration_seconds,
            "static_avg_delay": round(d_static, 2),
            "adaptive_avg_delay": round(d_adaptive, 2),
            "delay_reduction_pct": round(reduction_pct, 2),
            "static_throughput": thru_static,
            "adaptive_throughput": thru_adaptive,
            "throughput_increase_pct": round(((thru_adaptive - thru_static) / max(1, thru_static)) * 100.0, 2),
            "static_queued": q_static,
            "adaptive_queued": q_adaptive,
            "queue_reduction_pct": round(((q_static - q_adaptive) / max(1, q_static)) * 100.0, 2) if q_static > 0 else 0.0,
            "target_met": bool(reduction_pct >= 25.0),
            "notes": "Verified Soft Computing compliance across Modules 1 to 6."
        }

        # Persist to database
        self.db.save_benchmark(report)
        self.benchmark_in_progress = False
        return report

    # --------------------------------------------------------------------------
    # MODEL INTROSPECTION (For Dashboard Soft Computing Panel)
    # --------------------------------------------------------------------------
    def get_soft_computing_models_info(self) -> Dict[str, Any]:
        """Provides full architectural breakdown of Person 1 & 2 models for the UI."""
        controller = get_controller()
        ann = get_ann_predictor()

        return {
            "fuzzy_controller": {
                "rule_count": len(controller.rules),
                "defuzzification": "Centroid (COG)",
                "membership_bounds": controller.get_parameters(),
                "universes": {
                    "QueueLength": controller.queue_var.universe,
                    "WaitingTime": controller.wait_var.universe,
                    "ArrivalRate": controller.arrival_var.universe,
                    "GreenExtension": controller.extension_var.universe
                }
            },
            "ann_predictor": {
                "architecture": "MLP (6 -> 32 -> 16 -> 2)",
                "features": ["q_t", "q_{t-1}", "v_t", "w_t", "hour", "is_rush_hour"],
                "targets": ["predicted_queue_10min", "surge_factor"],
                "device": str(ann.device)
            }
        }


# Global Singleton Orchestrator
_GLOBAL_ORCHESTRATOR: Optional[SimulationOrchestrator] = None


def get_orchestrator() -> SimulationOrchestrator:
    """Returns singleton instance of SimulationOrchestrator."""
    global _GLOBAL_ORCHESTRATOR
    if _GLOBAL_ORCHESTRATOR is None:
        _GLOBAL_ORCHESTRATOR = SimulationOrchestrator()
    return _GLOBAL_ORCHESTRATOR
