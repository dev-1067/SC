"""
Smart Traffic Signal Control System (STCS)
Person 2 Deliverable: Traffic Simulation Environment (simulation_env.py)
Covers Simulation & Control Execution:
- 4-Way Crossroad Junction (North-South & East-West approaches)
- Car-Following Physics Engine (Intelligent Driver Model / Gap Maintenance)
- Traffic Light Actuators: RED, YELLOW, GREEN with Yellow Clearance Interval
- Dual Control Modes: Adaptive Soft Computing (Person 1 FLC) vs. Static Fixed-Time
- Headless (Backend / API / Benchmarking) & GUI (Pygame Interactive Visualizer) modes
- High-Frequency Telemetry Export for Person 3 (FastAPI & React Dashboard)
"""

import math
import os
import random
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Pygame is optional in headless mode if not display-attached, but installed and supported
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from fuzzy_engine import get_optimal_phase
from ann_predictor import get_ann_predictor


# ==============================================================================
# 1. VEHICLE SIMULATION AGENT
# ==============================================================================

class SimVehicle:
    """Individual vehicle with car-following dynamics and delay accounting."""

    def __init__(self, vehicle_id: int, approach: str, spawn_pos: float, speed: float = 6.0):
        self.vehicle_id = vehicle_id
        self.approach = approach  # "North", "South", "East", "West"
        self.speed = speed
        self.max_speed = random.uniform(6.5, 9.0)
        self.accel = 0.5
        self.decel = 0.8
        self.length = 26
        self.width = 16
        self.wait_time = 0.0
        self.total_travel_time = 0.0
        self.is_queued = False
        self.has_cleared_intersection = False

        # Palette for cars
        palette = [
            (210, 50, 50),   # Crimson
            (50, 110, 210),  # Royal Blue
            (220, 220, 225), # Silver
            (230, 180, 30),  # Amber Gold
            (40, 180, 110),  # Emerald
            (50, 50, 55)     # Charcoal
        ]
        self.color = random.choice(palette)

        # 1D position along approach axis (pixels from origin of lane)
        self.pos = spawn_pos

    def update_physics(self, lead_vehicle: Optional["SimVehicle"], light_state: str,
                       stop_line_pos: float, dt: float = 1.0):
        """
        Car-Following logic:
        Maintains safe following gap s0 behind lead vehicle, stops at RED/YELLOW lights,
        accelerates freely on GREEN.
        """
        self.total_travel_time += dt

        target_speed = self.max_speed
        dist_to_stop = stop_line_pos - self.pos

        # Check traffic light constraint if not past stop line
        if dist_to_stop > -5:
            if light_state == "RED":
                if dist_to_stop <= 0:
                    # Stopped at or slightly over stop line on red -> remain stopped
                    target_speed = 0.0
                elif dist_to_stop < 140:
                    # Brake smoothly to stop at stop line
                    target_speed = min(target_speed, max(0.0, (dist_to_stop - 8) * 0.08))
            elif light_state == "YELLOW":
                # Dilemma zone logic: stop if feasible, proceed if too close to stop safely
                if dist_to_stop > 60:
                    target_speed = min(target_speed, max(0.0, (dist_to_stop - 8) * 0.08))
                else:
                    # Clear intersection
                    target_speed = self.max_speed
        else:
            self.has_cleared_intersection = True

        # Check lead vehicle gap constraint
        if lead_vehicle is not None:
            dist_to_lead = (lead_vehicle.pos - lead_vehicle.length) - self.pos
            safe_gap = 14.0 + self.speed * 1.5
            if dist_to_lead < safe_gap:
                target_speed = min(target_speed, max(0.0, lead_vehicle.speed * 0.5))
            elif dist_to_lead < safe_gap * 2.0:
                target_speed = min(target_speed, lead_vehicle.speed)

        # Accelerate or decelerate towards target speed
        if self.speed < target_speed:
            self.speed = min(target_speed, self.speed + self.accel * dt)
        else:
            # Active responsive braking when close to lead vehicle or stop line
            brake_force = self.decel
            if lead_vehicle is not None and (lead_vehicle.pos - lead_vehicle.length - self.pos) < 35.0:
                brake_force = 0.8
            elif dist_to_stop < 50.0 and light_state in ("RED", "YELLOW"):
                brake_force = 0.8
            self.speed = max(target_speed, self.speed - brake_force * dt)

        # Advance position
        self.pos += self.speed * dt

        # Queue tracking
        if self.speed < 0.35 and not self.has_cleared_intersection:
            self.is_queued = True
            self.wait_time += dt
        else:
            self.is_queued = False


# ==============================================================================
# 2. 4-WAY INTERSECTION SIMULATION ENVIRONMENT
# ==============================================================================

class TrafficSimulation:
    """
    Stateful 4-Way Crossroad Traffic Simulation Environment.
    Manages vehicle generation, physics updates, signal actuations, telemetry extraction,
    and adaptive vs. static signal switching.
    """

    def __init__(
        self,
        simulation_size: int = 800,
        control_mode: str = "ADAPTIVE",  # "ADAPTIVE" or "STATIC_FIXED"
        headless: bool = True,
        random_seed: Optional[int] = 42
    ):
        self.size = simulation_size
        self.cx = simulation_size // 2
        self.cy = simulation_size // 2
        self.control_mode = control_mode
        self.headless = headless

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.road_width = 140
        self.half_road = self.road_width // 2

        # Geometry Coordinates
        # Stop lines (distance from screen origin for each approach)
        self.stop_lines = {
            "North": self.cy - self.half_road,
            "South": self.cy + self.half_road,
            "West":  self.cx - self.half_road,
            "East":  self.cx + self.half_road
        }

        # Lane vehicles lists
        self.vehicles: Dict[str, List[SimVehicle]] = {
            "North": [], "South": [], "East": [], "West": []
        }

        # Traffic Signal States: 0 = North-South Green, 1 = East-West Green
        self.active_phase = 0
        self.phase_names = {0: "North-South", 1: "East-West"}
        self.signals = {"NS": "GREEN", "EW": "RED"}

        # Signal Timers
        self.elapsed_in_phase = 0.0
        self.allocated_green = 30.0
        self.min_green_seconds = 10.0
        self.max_green_seconds = 60.0
        self.yellow_duration = 3.0
        self.in_yellow_transition = False
        self.pending_next_phase = 0
        self.emergency_preemption_active = False
        self.emergency_target_phase = 0

        # Arrival Flow Demands (vehicles per minute per approach)
        self.arrival_rates: Dict[str, float] = {
            "North": 4.5, "South": 4.5, "East": 5.0, "West": 5.0
        }

        # Historical & Diagnostic Metrics
        self.next_veh_id = 1
        self.sim_time = 0.0
        self.total_throughput = 0
        self.completed_delays: List[float] = []
        self.flc_decisions_history: List[Dict[str, Any]] = []

        # Pygame Rendering attributes
        self.screen = None
        self.clock = None
        self.font = None
        self.font_large = None
        if not self.headless and PYGAME_AVAILABLE:
            self._init_pygame_ui()

    def _init_pygame_ui(self):
        """Initializes Pygame display and fonts."""
        pygame.init()
        self.screen = pygame.display.set_mode((self.size, self.size))
        pygame.display.set_caption("Smart Traffic Signal Control System (STCS) - Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 14)
        self.font_large = pygame.font.SysFont("Arial", 20, bold=True)

    def set_control_mode(self, mode: str):
        """Switches between 'ADAPTIVE' (FLC) and 'STATIC_FIXED' modes."""
        assert mode in ("ADAPTIVE", "STATIC_FIXED", "MANUAL")
        self.control_mode = mode

    def spawn_vehicles(self, dt: float = 1.0):
        """Stochastically spawns vehicles on each lane based on arrival rates."""
        for approach in ["North", "South", "East", "West"]:
            # Probability of arrival in interval dt = (arrival_rate / 60) * dt
            p_arrival = (self.arrival_rates[approach] / 60.0) * dt
            if random.random() < p_arrival:
                # Ensure distance from last spawned vehicle
                can_spawn = True
                if self.vehicles[approach]:
                    trailing_car = self.vehicles[approach][-1]
                    if trailing_car.pos < 40:
                        can_spawn = False

                if can_spawn:
                    new_car = SimVehicle(self.next_veh_id, approach, spawn_pos=0.0)
                    self.vehicles[approach].append(new_car)
                    self.next_veh_id += 1

    def step(self, dt: float = 0.5) -> Dict[str, Any]:
        """
        Executes one physics & control iteration:
        1. Spawns vehicles
        2. Updates vehicle physics & stop lines
        3. Evaluates Traffic Controller (Adaptive FLC or Static Timer)
        4. Despawns departing vehicles & records metrics
        """
        self.sim_time += dt
        self.elapsed_in_phase += dt

        # Step 1: Spawn
        self.spawn_vehicles(dt)

        # Step 2: Update Vehicles
        for approach in ["North", "South", "East", "West"]:
            light_state = self.signals["NS"] if approach in ("North", "South") else self.signals["EW"]
            stop_pos = self.stop_lines[approach]

            # Invert coordinates for South and East approaches so physics is unified
            # For North & West: vehicles move forward from 0 towards stop line
            # For South & East: vehicles move backward from size towards stop line
            target_stop = stop_pos if approach in ("North", "West") else (self.size - stop_pos)

            for i, veh in enumerate(self.vehicles[approach]):
                lead = self.vehicles[approach][i - 1] if i > 0 else None
                veh.update_physics(lead, light_state, target_stop, dt)

        # Step 3: Remove completed vehicles
        for approach in ["North", "South", "East", "West"]:
            remaining = []
            for veh in self.vehicles[approach]:
                if veh.pos < self.size + 50:
                    remaining.append(veh)
                else:
                    self.total_throughput += 1
                    self.completed_delays.append(veh.wait_time)
            self.vehicles[approach] = remaining

        # Step 4: Traffic Signal Controller Logic
        self._manage_signal_cycle(dt)

        # Return full state snapshot
        return self.get_state()

    def _manage_signal_cycle(self, dt: float):
        """Controls signal transitions, yellow clearance, and phase switching."""
        if self.in_yellow_transition:
            # Handling 3s Yellow clearance interval
            if self.elapsed_in_phase >= self.yellow_duration:
                # Transition complete: activate green on new phase
                self.in_yellow_transition = False
                self.active_phase = self.pending_next_phase
                self.elapsed_in_phase = 0.0

                if self.active_phase == 0:
                    self.signals["NS"] = "GREEN"
                    self.signals["EW"] = "RED"
                else:
                    self.signals["NS"] = "RED"
                    self.signals["EW"] = "GREEN"
            return

        # Green phase execution: check if switch is requested
        should_switch = False
        next_phase = self.active_phase

        if getattr(self, "emergency_preemption_active", False):
            target = getattr(self, "emergency_target_phase", 0)
            if self.active_phase == target:
                self.elapsed_in_phase = 0.0  # Hold priority green for emergency vehicle
                return
            else:
                should_switch = True
                next_phase = target

        elif self.control_mode == "STATIC_FIXED":
            # Fixed 30s timer
            static_green_time = 30.0
            if self.elapsed_in_phase >= static_green_time:
                should_switch = True
                next_phase = 1 if self.active_phase == 0 else 0

        elif self.control_mode == "ADAPTIVE":
            # Soft Computing Decision Engine (Person 1 Hook)
            lane_metrics = self.get_lane_metrics_payload()
            lane_metrics["current_phase"] = self.active_phase
            lane_metrics["elapsed_green"] = self.elapsed_in_phase
            lane_metrics["min_green_seconds"] = self.min_green_seconds
            lane_metrics["max_green_seconds"] = self.max_green_seconds

            decision = get_optimal_phase(lane_metrics)
            self.allocated_green = decision["allocated_green_seconds"]
            should_switch = decision["should_switch"]
            next_phase = decision["next_phase_id"]

            if len(self.flc_decisions_history) > 20:
                self.flc_decisions_history.pop(0)
            self.flc_decisions_history.append(decision)

        # Execute switch via yellow safety interval
        if should_switch and next_phase != self.active_phase:
            self.in_yellow_transition = True
            self.pending_next_phase = next_phase
            self.elapsed_in_phase = 0.0
            # Set current green phase to YELLOW
            if self.active_phase == 0:
                self.signals["NS"] = "YELLOW"
                self.signals["EW"] = "RED"
            else:
                self.signals["NS"] = "RED"
                self.signals["EW"] = "YELLOW"

    def get_lane_metrics_payload(self) -> Dict[str, Any]:
        """Calculates crisp lane metrics formatted for Person 1 (FLC)."""
        metrics = {}
        for app in ["North", "South", "East", "West"]:
            vehs = self.vehicles[app]
            q = sum(1 for v in vehs if v.is_queued)
            wait = max([v.wait_time for v in vehs if v.is_queued] or [0.0])
            metrics[app] = {
                "queue_length": q,
                "wait_time": round(wait, 1),
                "arrival_rate": self.arrival_rates[app]
            }

        # Paired phases 0 (NS) and 1 (EW)
        phases = {
            0: {
                "queue_length": metrics["North"]["queue_length"] + metrics["South"]["queue_length"],
                "wait_time": max(metrics["North"]["wait_time"], metrics["South"]["wait_time"]),
                "arrival_rate": metrics["North"]["arrival_rate"] + metrics["South"]["arrival_rate"],
                "name": "North-South"
            },
            1: {
                "queue_length": metrics["East"]["queue_length"] + metrics["West"]["queue_length"],
                "wait_time": max(metrics["East"]["wait_time"], metrics["West"]["wait_time"]),
                "arrival_rate": metrics["East"]["arrival_rate"] + metrics["West"]["arrival_rate"],
                "name": "East-West"
            }
        }

        # Module 2 & 3: Flow forecasting via ANN Predictor
        try:
            predictor = get_ann_predictor()
            hour = (12.0 + self.sim_time / 3600.0) % 24.0
            for p_id in (0, 1):
                q_cur = float(phases[p_id]["queue_length"])
                attr = f"_prev_q_{p_id}"
                q_prev = getattr(self, attr, q_cur)
                setattr(self, attr, q_cur)
                pred = predictor.predict_flow(
                    q_t=q_cur,
                    q_t_prev=q_prev,
                    v_t=float(phases[p_id]["arrival_rate"]),
                    w_t=float(phases[p_id]["wait_time"]),
                    hour_of_day=hour
                )
                phases[p_id]["predicted_queue_10min"] = pred["predicted_queue_10min"]
                phases[p_id]["surge_factor"] = pred["surge_factor"]
                phases[p_id]["is_surge_imminent"] = pred["is_surge_imminent"]
        except Exception:
            for p_id in (0, 1):
                phases[p_id]["predicted_queue_10min"] = float(phases[p_id]["queue_length"])
                phases[p_id]["surge_factor"] = 0.0
                phases[p_id]["is_surge_imminent"] = False

        return {"phases": phases, "lanes": metrics}

    def get_state(self) -> Dict[str, Any]:
        """Complete JSON-serializable telemetry state for Person 3 API/WebSockets."""
        lane_payload = self.get_lane_metrics_payload()
        all_delays = self.completed_delays + [v.wait_time for v_list in self.vehicles.values() for v in v_list]
        avg_delay = float(np.mean(all_delays)) if all_delays else 0.0
        active_veh_count = sum(len(v) for v in self.vehicles.values())
        total_queued = sum(lane_payload["phases"][p]["queue_length"] for p in (0, 1))

        # Vehicle position coordinates for UI rendering
        vehicle_coords = []
        for app, v_list in self.vehicles.items():
            for v in v_list:
                px, py = self._get_vehicle_screen_coords(app, v.pos)
                vehicle_coords.append({
                    "id": v.vehicle_id, "approach": app,
                    "x": round(px, 1), "y": round(py, 1),
                    "speed": round(v.speed, 2), "is_queued": v.is_queued,
                    "wait_time": round(v.wait_time, 1),
                    "color": v.color
                })

        return {
            "sim_time": round(self.sim_time, 1),
            "control_mode": self.control_mode,
            "active_phase": self.active_phase,
            "phase_name": self.phase_names[self.active_phase],
            "signal_states": self.signals,
            "elapsed_in_phase": round(self.elapsed_in_phase, 1),
            "allocated_green": round(self.allocated_green, 1),
            "in_yellow_transition": self.in_yellow_transition,
            "phases": lane_payload["phases"],
            "lanes": lane_payload["lanes"],
            "active_vehicle_count": active_veh_count,
            "total_queued": total_queued,
            "total_throughput": self.total_throughput,
            "average_delay_seconds": round(avg_delay, 1),
            "ann_forecast": {
                str(p): {
                    "predicted_queue_10min": lane_payload["phases"][p].get("predicted_queue_10min", 0.0),
                    "surge_factor": lane_payload["phases"][p].get("surge_factor", 0.0),
                    "is_surge_imminent": lane_payload["phases"][p].get("is_surge_imminent", False)
                } for p in (0, 1)
            },
            "vehicles": vehicle_coords,
            "latest_flc_decision": self.flc_decisions_history[-1] if self.flc_decisions_history else None
        }

    def _get_vehicle_screen_coords(self, approach: str, pos: float) -> Tuple[float, float]:
        """Translates 1D lane progress pos into 2D (x, y) screen coordinates."""
        lane_offset = 25
        if approach == "North":
            return float(self.cx - lane_offset), float(pos)
        elif approach == "South":
            return float(self.cx + lane_offset), float(self.size - pos)
        elif approach == "West":
            return float(pos), float(self.cy + lane_offset)
        else: # East
            return float(self.size - pos), float(self.cy - lane_offset)

    def render_pygame(self):
        """Renders 60 FPS graphical intersection view when GUI mode is active."""
        if self.headless or not PYGAME_AVAILABLE or self.screen is None:
            return

        # Background grass
        self.screen.fill((45, 95, 50))

        # Roads
        road_col = (40, 40, 45)
        # North-South road
        pygame.draw.rect(self.screen, road_col, (self.cx - self.half_road, 0, self.road_width, self.size))
        # East-West road
        pygame.draw.rect(self.screen, road_col, (0, self.cy - self.half_road, self.size, self.road_width))

        # Yellow center dividers
        divider_col = (235, 195, 30)
        # Vertical divider
        for y in range(0, self.size, 30):
            if not (self.cy - self.half_road < y < self.cy + self.half_road):
                pygame.draw.line(self.screen, divider_col, (self.cx, y), (self.cx, y + 16), 2)
        # Horizontal divider
        for x in range(0, self.size, 30):
            if not (self.cx - self.half_road < x < self.cx + self.half_road):
                pygame.draw.line(self.screen, divider_col, (x, self.cy), (x + 16, self.cy), 2)

        # White Stop lines
        white = (245, 245, 250)
        pygame.draw.line(self.screen, white, (self.cx - self.half_road, self.cy - self.half_road), (self.cx, self.cy - self.half_road), 4)
        pygame.draw.line(self.screen, white, (self.cx, self.cy + self.half_road), (self.cx + self.half_road, self.cy + self.half_road), 4)
        pygame.draw.line(self.screen, white, (self.cx - self.half_road, self.cy), (self.cx - self.half_road, self.cy + self.half_road), 4)
        pygame.draw.line(self.screen, white, (self.cx + self.half_road, self.cy - self.half_road), (self.cx + self.half_road, self.cy), 4)

        # Draw Traffic Lights
        self._draw_traffic_lights()

        # Draw Vehicles
        state = self.get_state()
        for v in state["vehicles"]:
            app = v["approach"]
            x, y = int(v["x"]), int(v["y"])
            col = v["color"]
            if app in ("North", "South"):
                pygame.draw.rect(self.screen, col, (x - 8, y - 13, 16, 26), border_radius=3)
            else:
                pygame.draw.rect(self.screen, col, (x - 13, y - 8, 26, 16), border_radius=3)

            # Brake lights if stopped
            if v["is_queued"]:
                brake_col = (255, 30, 30)
                if app == "North":
                    pygame.draw.circle(self.screen, brake_col, (x - 5, y - 11), 2)
                    pygame.draw.circle(self.screen, brake_col, (x + 5, y - 11), 2)
                elif app == "South":
                    pygame.draw.circle(self.screen, brake_col, (x - 5, y + 11), 2)
                    pygame.draw.circle(self.screen, brake_col, (x + 5, y + 11), 2)

        # Draw HUD Panel
        self._draw_hud(state)

        pygame.display.flip()
        self.clock.tick(60)

    def _draw_traffic_lights(self):
        """Draws glowing traffic light fixtures."""
        # NS light (North corner)
        ns_col = (50, 220, 50) if self.signals["NS"] == "GREEN" else (
            (240, 200, 30) if self.signals["NS"] == "YELLOW" else (240, 40, 40)
        )
        # EW light (West corner)
        ew_col = (50, 220, 50) if self.signals["EW"] == "GREEN" else (
            (240, 200, 30) if self.signals["EW"] == "YELLOW" else (240, 40, 40)
        )

        # Light housing rects
        pygame.draw.circle(self.screen, (20, 20, 20), (self.cx - self.half_road - 20, self.cy - self.half_road - 20), 16)
        pygame.draw.circle(self.screen, ns_col, (self.cx - self.half_road - 20, self.cy - self.half_road - 20), 12)

        pygame.draw.circle(self.screen, (20, 20, 20), (self.cx + self.half_road + 20, self.cy + self.half_road + 20), 16)
        pygame.draw.circle(self.screen, ew_col, (self.cx + self.half_road + 20, self.cy + self.half_road + 20), 12)

    def _draw_hud(self, state: Dict[str, Any]):
        """Draws top HUD status card."""
        hud_bg = pygame.Surface((self.size - 40, 75), pygame.SRCALPHA)
        hud_bg.fill((20, 24, 30, 210))
        self.screen.blit(hud_bg, (20, 15))

        # Mode and status text
        mode_text = self.font_large.render(f"STCS Control Mode: {state['control_mode']}", True, (100, 220, 255))
        self.screen.blit(mode_text, (35, 24))

        status_text = self.font.render(
            f"Active: {state['phase_name']} ({state['signal_states']['NS']}/{state['signal_states']['EW']}) | "
            f"Phase Time: {state['elapsed_in_phase']}s / {state['allocated_green']}s | "
            f"Queued: {state['total_queued']} | Throughput: {state['total_throughput']} veh",
            True, (230, 230, 235)
        )
        self.screen.blit(status_text, (35, 54))


# ==============================================================================
# SCRIPT ENTRYPOINT FOR STANDALONE SIMULATION RUNNER
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Running 4-Way Traffic Simulation Environment (Person 2)")
    print("Mode: ADAPTIVE (FLC Actuation + Pygame Engine)")
    print("=" * 70)

    sim = TrafficSimulation(control_mode="ADAPTIVE", headless=True)
    for step_num in range(60):
        state = sim.step(dt=0.5)
        if step_num % 15 == 0:
            print(f"Time {state['sim_time']:5.1f}s | Active: {state['phase_name']:<11} | "
                  f"Signals: NS={state['signal_states']['NS']:<6} EW={state['signal_states']['EW']:<6} | "
                  f"Queues: NS={state['phases'][0]['queue_length']:2d}, EW={state['phases'][1]['queue_length']:2d} | "
                  f"Throughput: {state['total_throughput']} veh")

    print("\nSimulation successfully completed 60 steps headless!")
