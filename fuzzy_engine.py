"""
Smart Traffic Signal Control System (STCS)
Person 1 Deliverable: Fuzzy Inference System (FLC Engine)
Covers Syllabus Modules 4 & 5:
- Fuzzy Set Theory: Triangular & Trapezoidal Membership Functions, Fuzzification
- Mamdani Fuzzy Inference Engine: 36 IF-THEN Rules, Min T-Norm, Max S-Norm
- Defuzzification: Center of Gravity / Centroid Method (COG)
- Integration Hook: get_optimal_phase(lane_metrics)
"""

import json
import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


# ==============================================================================
# 1. CORE MEMBERSHIP FUNCTIONS (Module 4)
# ==============================================================================

def trimf(x: np.ndarray | float, params: List[float] | Tuple[float, float, float]) -> np.ndarray | float:
    """
    Triangular Membership Function: trimf(x; a, b, c)
    mu(x) = max(0, min((x-a)/(b-a), (c-x)/(c-b)))
    """
    a, b, c = params
    assert a <= b <= c, f"Triangular MF requires a <= b <= c, got {a}, {b}, {c}"
    
    is_scalar = np.isscalar(x)
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr)
    
    # Left slope
    if b > a:
        left_mask = (a <= x_arr) & (x_arr <= b)
        y[left_mask] = (x_arr[left_mask] - a) / (b - a)
    else:
        y[x_arr == b] = 1.0

    # Right slope
    if c > b:
        right_mask = (b < x_arr) & (x_arr <= c)
        y[right_mask] = (c - x_arr[right_mask]) / (c - b)
        
    y = np.clip(y, 0.0, 1.0)
    return float(y.item()) if is_scalar else y


def trapmf(x: np.ndarray | float, params: List[float] | Tuple[float, float, float, float]) -> np.ndarray | float:
    """
    Trapezoidal Membership Function: trapmf(x; a, b, c, d)
    Handles open-left shoulder if a == b and open-right shoulder if c == d.
    """
    a, b, c, d = params
    assert a <= b <= c <= d, f"Trapezoidal MF requires a <= b <= c <= d, got {a}, {b}, {c}, {d}"
    
    is_scalar = np.isscalar(x)
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr)
    
    # Left shoulder / Left slope
    if b > a:
        left_mask = (a <= x_arr) & (x_arr < b)
        y[left_mask] = (x_arr[left_mask] - a) / (b - a)
    else:
        # Open left shoulder
        y[x_arr <= a] = 1.0

    # Top plateau
    plateau_mask = (b <= x_arr) & (x_arr <= c)
    y[plateau_mask] = 1.0

    # Right slope / Right shoulder
    if d > c:
        right_mask = (c < x_arr) & (x_arr <= d)
        y[right_mask] = (d - x_arr[right_mask]) / (d - c)
    else:
        # Open right shoulder
        y[x_arr >= d] = 1.0

    y = np.clip(y, 0.0, 1.0)
    return float(y.item()) if is_scalar else y


# ==============================================================================
# 2. LINGUISTIC VARIABLES & FUZZY SET DEFINITIONS
# ==============================================================================

class LinguisticVariable:
    """Represents an input or output linguistic variable with defined universe and terms."""
    def __init__(self, name: str, universe: Tuple[float, float], unit: str = ""):
        self.name = name
        self.universe = universe  # (min, max)
        self.unit = unit
        self.terms: Dict[str, Dict[str, Any]] = {}

    def add_term(self, term_name: str, mf_type: str, params: List[float]):
        """Register a linguistic term with either triangular or trapezoidal MF."""
        assert mf_type in ('tri', 'trap'), "mf_type must be 'tri' or 'trap'"
        self.terms[term_name] = {
            'type': mf_type,
            'params': list(params)
        }

    def evaluate(self, x: float) -> Dict[str, float]:
        """Compute membership degree for each term for a given crisp value."""
        x_clamped = np.clip(x, self.universe[0], self.universe[1])
        degrees = {}
        for term, config in self.terms.items():
            if config['type'] == 'tri':
                degrees[term] = float(trimf(x_clamped, config['params']))
            else:
                degrees[term] = float(trapmf(x_clamped, config['params']))
        return degrees

    def evaluate_curve(self, x_grid: np.ndarray, term_name: str) -> np.ndarray:
        """Compute membership values across a 1D grid for plotting/defuzzification."""
        config = self.terms[term_name]
        if config['type'] == 'tri':
            return trimf(x_grid, config['params'])
        else:
            return trapmf(x_grid, config['params'])


# ==============================================================================
# 3. MAMDANI FUZZY RULE & INFERENCE ENGINE (Module 5)
# ==============================================================================

class FuzzyRule:
    """
    Mamdani IF-THEN Rule:
    IF Queue is <Q> AND Wait is <W> AND Arrival is <V> THEN GreenExtension is <T> (weight w)
    """
    def __init__(self, rule_id: int, queue_term: str, wait_term: str, arrival_term: str,
                 output_term: str, weight: float = 1.0, rationale: str = ""):
        self.rule_id = rule_id
        self.queue_term = queue_term
        self.wait_term = wait_term
        self.arrival_term = arrival_term
        self.output_term = output_term
        self.weight = weight
        self.rationale = rationale

    def evaluate_antecedent(self, q_degrees: Dict[str, float],
                            w_degrees: Dict[str, float],
                            v_degrees: Dict[str, float]) -> float:
        """Mamdani Min T-Norm for fuzzy AND conjunction."""
        deg_q = q_degrees.get(self.queue_term, 0.0)
        deg_w = w_degrees.get(self.wait_term, 0.0)
        deg_v = v_degrees.get(self.arrival_term, 0.0)
        firing_strength = min(deg_q, deg_w, deg_v) * self.weight
        return firing_strength

    def __repr__(self):
        return (f"Rule #{self.rule_id:02d}: IF Q is {self.queue_term} AND W is {self.wait_term} "
                f"AND Arr is {self.arrival_term} THEN Extension is {self.output_term}")


class FuzzyTrafficController:
    """
    Complete Mamdani Fuzzy Logic Controller for Smart Traffic Signal Extension.
    - Inputs:
      1. Queue Length (0 - 50 vehicles)
      2. Waiting Time (0 - 180 seconds)
      3. Arrival Rate (0 - 10 vehicles/min)
    - Output:
      Green Time Extension (10 - 60 seconds)
    - Defuzzification: Centroid Method (Center of Gravity)
    """

    def __init__(self, custom_params: Optional[Dict[str, Any]] = None):
        self.resolution = 500  # Grid resolution for numerical integration
        self._init_variables()
        self._init_rules()
        if custom_params:
            self.load_parameters(custom_params)

    def _init_variables(self):
        # 1. Queue Length Q in [0, 50] (Ruspini Partition)
        self.queue_var = LinguisticVariable("QueueLength", (0.0, 50.0), "vehicles")
        self.queue_var.add_term("Short", "trap", [0.0, 0.0, 8.0, 20.0])
        self.queue_var.add_term("Medium", "tri", [8.0, 20.0, 35.0])
        self.queue_var.add_term("Long", "tri", [20.0, 35.0, 45.0])
        self.queue_var.add_term("VeryLong", "trap", [35.0, 45.0, 50.0, 50.0])

        # 2. Waiting Time W in [0, 180] (Ruspini Partition)
        self.wait_var = LinguisticVariable("WaitingTime", (0.0, 180.0), "seconds")
        self.wait_var.add_term("Low", "trap", [0.0, 0.0, 40.0, 90.0])
        self.wait_var.add_term("Normal", "tri", [40.0, 90.0, 140.0])
        self.wait_var.add_term("Critical", "trap", [90.0, 140.0, 180.0, 180.0])

        # 3. Arrival Rate Delta V in [0, 10] (Ruspini Partition)
        self.arrival_var = LinguisticVariable("ArrivalRate", (0.0, 10.0), "veh/min")
        self.arrival_var.add_term("Low", "trap", [0.0, 0.0, 2.0, 5.0])
        self.arrival_var.add_term("Medium", "tri", [2.0, 5.0, 8.0])
        self.arrival_var.add_term("High", "trap", [5.0, 8.0, 10.0, 10.0])

        # 4. Output: Green Signal Time Extension in [10, 60] (Ruspini Partition)
        self.extension_var = LinguisticVariable("GreenExtension", (10.0, 60.0), "seconds")
        self.extension_var.add_term("VeryShort", "trap", [10.0, 10.0, 15.0, 22.5])
        self.extension_var.add_term("Short", "tri", [15.0, 22.5, 35.0])
        self.extension_var.add_term("Medium", "tri", [22.5, 35.0, 47.5])
        self.extension_var.add_term("Long", "tri", [35.0, 47.5, 55.0])
        self.extension_var.add_term("VeryLong", "trap", [47.5, 55.0, 60.0, 60.0])

    def _init_rules(self):
        """
        Build complete 36-rule Mamdani Rule Base (4 Queue x 3 Wait x 3 Arrival).
        Provides comprehensive, monotonic soft computing control under all conditions.
        """
        self.rules: List[FuzzyRule] = []
        rule_spec = [
            # Short Queue (4 rules x 3 Wait x 3 Arrival = 9 rules)
            (1,  "Short",    "Low",      "Low",    "VeryShort", "Minimal queue and low demand"),
            (2,  "Short",    "Low",      "Medium", "VeryShort", "Minor arriving platoon"),
            (3,  "Short",    "Low",      "High",   "Short",     "High incoming burst needs brief buffer"),
            (4,  "Short",    "Normal",   "Low",    "Short",     "Moderate wait warranting signal clearance"),
            (5,  "Short",    "Normal",   "Medium", "Short",     "Balanced low-volume clearing"),
            (6,  "Short",    "Normal",   "High",   "Medium",    "Arriving traffic benefits from progression"),
            (7,  "Short",    "Critical", "Low",    "Medium",    "Stale wait time priority clearance"),
            (8,  "Short",    "Critical", "Medium", "Medium",    "Critical wait plus moderate arrivals"),
            (9,  "Short",    "Critical", "High",   "Long",      "Critical wait facing high incoming flow"),

            # Medium Queue (9 rules)
            (10, "Medium",   "Low",      "Low",    "Short",     "Clear standard batch of cars"),
            (11, "Medium",   "Low",      "Medium", "Medium",    "Sustain green for steady inflow"),
            (12, "Medium",   "Low",      "High",   "Long",      "Incoming surge behind medium queue"),
            (13, "Medium",   "Normal",   "Low",    "Medium",    "Standard traffic cycle"),
            (14, "Medium",   "Normal",   "Medium", "Medium",    "Typical urban midday flow"),
            (15, "Medium",   "Normal",   "High",   "Long",      "Flow building up, extend green"),
            (16, "Medium",   "Critical", "Low",    "Long",      "Prolonged wait on medium queue"),
            (17, "Medium",   "Critical", "Medium", "Long",      "Heavy priority clearance needed"),
            (18, "Medium",   "Critical", "High",   "VeryLong",  "Near bottleneck condition"),

            # Long Queue (9 rules)
            (19, "Long",     "Low",      "Low",    "Medium",    "Clear backed up queue"),
            (20, "Long",     "Low",      "Medium", "Long",      "High density discharge with flow"),
            (21, "Long",     "Low",      "High",   "Long",      "Major platoon handling"),
            (22, "Long",     "Normal",   "Low",    "Long",      "High volume clearance priority"),
            (23, "Long",     "Normal",   "Medium", "Long",      "Heavy persistent corridor flow"),
            (24, "Long",     "Normal",   "High",   "VeryLong",  "Congestion mitigation extension"),
            (25, "Long",     "Critical", "Low",    "VeryLong",  "Critical starvation prevention"),
            (26, "Long",     "Critical", "Medium", "VeryLong",  "Severe congestion release"),
            (27, "Long",     "Critical", "High",   "VeryLong",  "Maximum throughput emergency"),

            # Very Long Queue (9 rules)
            (28, "VeryLong", "Low",      "Low",    "Long",      "Sudden jam clearance"),
            (29, "VeryLong", "Low",      "Medium", "VeryLong",  "Spillback prevention"),
            (30, "VeryLong", "Low",      "High",   "VeryLong",  "Heavy rush hour inflow mitigation"),
            (31, "VeryLong", "Normal",   "Low",    "VeryLong",  "High backlog discharge"),
            (32, "VeryLong", "Normal",   "Medium", "VeryLong",  "Saturation recovery phase"),
            (33, "VeryLong", "Normal",   "High",   "VeryLong",  "Peak congestion maximum extension"),
            (34, "VeryLong", "Critical", "Low",    "VeryLong",  "Severe blockage immediate relief"),
            (35, "VeryLong", "Critical", "Medium", "VeryLong",  "Intersection gridlock defense"),
            (36, "VeryLong", "Critical", "High",   "VeryLong",  "Emergency max green allocation")
        ]

        for r_id, q_t, w_t, a_t, o_t, rat in rule_spec:
            self.rules.append(FuzzyRule(r_id, q_t, w_t, a_t, o_t, 1.0, rat))

    def get_parameters(self) -> Dict[str, Any]:
        """Export current membership parameters (useful for GA optimization)."""
        return {
            "QueueLength": {k: v['params'] for k, v in self.queue_var.terms.items()},
            "WaitingTime": {k: v['params'] for k, v in self.wait_var.terms.items()},
            "ArrivalRate": {k: v['params'] for k, v in self.arrival_var.terms.items()},
            "GreenExtension": {k: v['params'] for k, v in self.extension_var.terms.items()}
        }

    def load_parameters(self, params: Dict[str, Any]):
        """Load optimized membership parameters from dict or GA chromosome."""
        var_map = {
            "QueueLength": self.queue_var,
            "WaitingTime": self.wait_var,
            "ArrivalRate": self.arrival_var,
            "GreenExtension": self.extension_var
        }
        for var_name, terms_data in params.items():
            if var_name in var_map:
                for term_name, p_list in terms_data.items():
                    if term_name in var_map[var_name].terms:
                        var_map[var_name].terms[term_name]['params'] = list(p_list)

    # --------------------------------------------------------------------------
    # INFERENCE & DEFUZZIFICATION (Centroid COG Method)
    # --------------------------------------------------------------------------
    def calculate_green_extension(self, queue_length: float, waiting_time: float,
                                  arrival_rate: float) -> Dict[str, Any]:
        """
        Executes complete Mamdani Fuzzy Inference:
        1. Fuzzification of input values
        2. Rule evaluation (Min T-Norm)
        3. Implication & Aggregation (Max S-Norm)
        4. Defuzzification via Centroid Method:
           z* = (integral z * mu(z) dz) / (integral mu(z) dz)
        """
        # Robust input sanitization with universe clamping
        try:
            q_clean = float(queue_length)
            if np.isnan(q_clean) or np.isinf(q_clean):
                q_clean = 0.0
            else:
                q_clean = float(np.clip(q_clean, 0.0, 50.0))
        except (ValueError, TypeError):
            q_clean = 0.0

        try:
            w_clean = float(waiting_time)
            if np.isnan(w_clean) or np.isinf(w_clean):
                w_clean = 0.0
            else:
                w_clean = float(np.clip(w_clean, 0.0, 180.0))
        except (ValueError, TypeError):
            w_clean = 0.0

        try:
            v_clean = float(arrival_rate)
            if np.isnan(v_clean) or np.isinf(v_clean):
                v_clean = 0.0
            else:
                v_clean = float(np.clip(v_clean, 0.0, 10.0))
        except (ValueError, TypeError):
            v_clean = 0.0

        # Step 1: Fuzzify inputs
        q_deg = self.queue_var.evaluate(q_clean)
        w_deg = self.wait_var.evaluate(w_clean)
        v_deg = self.arrival_var.evaluate(v_clean)

        # Step 2: Evaluate rules
        active_rules = []
        consequent_weights: Dict[str, float] = {t: 0.0 for t in self.extension_var.terms.keys()}

        for rule in self.rules:
            firing_strength = rule.evaluate_antecedent(q_deg, w_deg, v_deg)
            if firing_strength > 1e-4:
                active_rules.append({
                    "rule_id": rule.rule_id,
                    "description": str(rule),
                    "firing_strength": round(float(firing_strength), 4),
                    "consequent": rule.output_term,
                    "rationale": rule.rationale
                })
                # Max aggregation for rules with identical consequents
                if firing_strength > consequent_weights[rule.output_term]:
                    consequent_weights[rule.output_term] = firing_strength

        # Sort active rules by firing strength descending
        active_rules.sort(key=lambda r: r["firing_strength"], reverse=True)

        # Step 3: Implication & Aggregation across continuous universe
        z_min, z_max = self.extension_var.universe
        z_grid = np.linspace(z_min, z_max, self.resolution)
        aggregated_mf = np.zeros_like(z_grid)

        for term, weight in consequent_weights.items():
            if weight > 0:
                # Mamdani implication (min clipping)
                mf_curve = self.extension_var.evaluate_curve(z_grid, term)
                clipped = np.minimum(mf_curve, weight)
                # Max S-norm aggregation
                aggregated_mf = np.maximum(aggregated_mf, clipped)

        # Step 4: Centroid Defuzzification (Center of Gravity)
        numerator = np.sum(z_grid * aggregated_mf)
        denominator = np.sum(aggregated_mf)

        if denominator > 1e-6:
            crisp_extension = float(numerator / denominator)
        else:
            # Fallback to center of output universe if no rules fired
            crisp_extension = (z_min + z_max) / 2.0

        # Enforce hard boundaries [10.0, 60.0] as mandated by spec
        crisp_extension = float(np.clip(crisp_extension, 10.0, 60.0))

        return {
            "green_extension_seconds": round(crisp_extension, 2),
            "input_metrics": {
                "queue_length": q_clean,
                "waiting_time": w_clean,
                "arrival_rate": v_clean
            },
            "fuzzified_inputs": {
                "QueueLength": {k: round(v, 4) for k, v in q_deg.items()},
                "WaitingTime": {k: round(v, 4) for k, v in w_deg.items()},
                "ArrivalRate": {k: round(v, 4) for k, v in v_deg.items()}
            },
            "consequent_weights": {k: round(v, 4) for k, v in consequent_weights.items()},
            "active_rules_count": len(active_rules),
            "active_rules": active_rules[:10],  # Top 10 for dashboard clarity
            "defuzzification_method": "Centroid (COG)",
            "output_bounds": [10.0, 60.0]
        }


# Global singleton instance for easy import across modules
_GLOBAL_CONTROLLER: Optional[FuzzyTrafficController] = None


def get_controller(param_file: Optional[str] = None, reload: bool = False) -> FuzzyTrafficController:
    """Returns or initializes the singleton FuzzyTrafficController."""
    global _GLOBAL_CONTROLLER
    if _GLOBAL_CONTROLLER is None or reload or param_file is not None:
        custom_params = None
        # Auto-load GA-optimized parameters if present
        default_param_path = os.path.join(os.path.dirname(__file__), "optimized_fuzzy_params.json")
        path_to_check = param_file or default_param_path
        if os.path.exists(path_to_check):
            try:
                with open(path_to_check, "r") as f:
                    custom_params = json.load(f)
            except Exception:
                pass
        _GLOBAL_CONTROLLER = FuzzyTrafficController(custom_params)
    return _GLOBAL_CONTROLLER


# ==============================================================================
# 4. UNIFIED INTEGRATION HOOK (Mandated by Spec Section 6 - Person 1)
# ==============================================================================

def get_optimal_phase(lane_metrics: dict) -> dict:
    """
    Unified Integration Hook for Person 2 (Simulation) & Person 3 (API/Dashboard).
    
    Signature mandated by spec:
        def get_optimal_phase(lane_metrics: dict) -> dict:
            '''Returns next phase ID and allocated green light duration in seconds.'''

    Supports flexible input formats:
    Case A: Standard Phase Dict
        lane_metrics = {
            "current_phase": 0,
            "elapsed_green": 15,
            "phases": {
                0: {"queue_length": 18, "wait_time": 45, "arrival_rate": 4.5, "name": "North-South"},
                1: {"queue_length": 32, "wait_time": 110, "arrival_rate": 7.2, "name": "East-West"}
            }
        }
    Case B: Direct Lane Metrics Dict
        lane_metrics = {
            "current_phase": 0,
            "lanes": {
                0: {"queue": 15, "wait": 40, "arrival": 3.0},
                1: {"queue": 25, "wait": 95, "arrival": 6.5}
            }
        }
    Case C: Single Approach Direct Evaluation
        lane_metrics = {"queue_length": 22, "wait_time": 65, "arrival_rate": 5.0}

    Returns:
        dict: {
            "next_phase_id": int | str,
            "allocated_green_seconds": float (10.0 - 60.0),
            "should_switch": bool,
            "phase_evaluations": dict,
            "decision_rationale": str,
            "active_rules": list,
            "fuzzy_details": dict
        }
    """
    # Defensive validation: handle None or malformed lane_metrics payload gracefully
    if not isinstance(lane_metrics, dict):
        lane_metrics = {}

    controller = get_controller()

    # Normalize inputs
    current_phase = lane_metrics.get("current_phase", 0)
    elapsed_green = float(lane_metrics.get("elapsed_green", 0.0))
    min_green_guarantee = float(lane_metrics.get("min_green_seconds", 10.0))
    max_green_cap = float(lane_metrics.get("max_green_seconds", 60.0))

    # Detect format
    phases_dict = {}
    if "phases" in lane_metrics and isinstance(lane_metrics["phases"], dict):
        phases_dict = lane_metrics["phases"]
    elif "lanes" in lane_metrics and isinstance(lane_metrics["lanes"], dict):
        phases_dict = lane_metrics["lanes"]
    elif "queue_length" in lane_metrics:
        # Single direct evaluation
        eval_res = controller.calculate_green_extension(
            queue_length=lane_metrics.get("queue_length", 0.0),
            waiting_time=lane_metrics.get("wait_time", 0.0),
            arrival_rate=lane_metrics.get("arrival_rate", 0.0)
        )
        return {
            "next_phase_id": current_phase,
            "allocated_green_seconds": eval_res["green_extension_seconds"],
            "should_switch": False,
            "decision_rationale": "Single approach evaluation via Mamdani FLC.",
            "active_rules": eval_res["active_rules"],
            "fuzzy_details": eval_res
        }
    else:
        # Default fallback 2-phase setup
        phases_dict = {
            0: {"queue_length": 0.0, "wait_time": 0.0, "arrival_rate": 0.0, "name": "North-South"},
            1: {"queue_length": 0.0, "wait_time": 0.0, "arrival_rate": 0.0, "name": "East-West"}
        }

    # Evaluate all candidate phases through FLC
    evaluations: Dict[Any, Dict[str, Any]] = {}
    urgency_scores: Dict[Any, float] = {}

    for p_id, p_data in phases_dict.items():
        q = float(p_data.get("queue_length", p_data.get("queue", 0.0)))
        w = float(p_data.get("wait_time", p_data.get("wait", 0.0)))
        arr = float(p_data.get("arrival_rate", p_data.get("arrival", 0.0)))

        flc_out = controller.calculate_green_extension(q, w, arr)
        evaluations[p_id] = flc_out
        
        # Soft Computing Urgency Metric:
        # Base urgency is the fuzzy recommended green duration, boosted by wait starvation and ANN surge forecast
        starvation_boost = 1.0 + (max(0.0, w - 90.0) / 100.0)
        surge_boost = 1.0 + (float(p_data.get("surge_factor", 0.0)) * 0.35)
        urgency = flc_out["green_extension_seconds"] * starvation_boost * surge_boost
        urgency_scores[p_id] = urgency

    # Phase Arbitration Logic
    # Match current_phase regardless of whether keys are int, str, or named approaches:
    if current_phase not in urgency_scores:
        matched = False
        # 1. Try matching string representations
        for k in urgency_scores:
            if str(k) == str(current_phase):
                current_phase = k
                matched = True
                break
        # 2. If integer index (e.g. 0, 1) but keys are named ("North", "East"):
        if not matched and isinstance(current_phase, int):
            keys_list = list(urgency_scores.keys())
            if 0 <= current_phase < len(keys_list):
                current_phase = keys_list[current_phase]
                matched = True
        # 3. Fallback to first available phase key
        if not matched and urgency_scores:
            current_phase = list(urgency_scores.keys())[0]

    curr_urgency = urgency_scores.get(current_phase, 0.0)
    curr_eval = evaluations.get(current_phase, list(evaluations.values())[0] if evaluations else {})
    
    # Find competitor with highest urgency
    other_phases = {k: v for k, v in urgency_scores.items() if k != current_phase}
    if other_phases:
        best_contender_id = max(other_phases, key=other_phases.get)
        contender_urgency = other_phases[best_contender_id]
    else:
        best_contender_id = current_phase
        contender_urgency = 0.0

    # Decision: should we extend current phase or switch?
    should_switch = False
    decision_rationale = ""
    next_phase_id = current_phase

    if elapsed_green < min_green_guarantee:
        # Must respect minimum green time for driver safety
        should_switch = False
        next_phase_id = current_phase
        allocated_green = max(min_green_guarantee, curr_eval.get("green_extension_seconds", min_green_guarantee))
        decision_rationale = f"Maintaining current phase {current_phase}: minimum green ({min_green_guarantee}s) not yet reached."
    elif elapsed_green >= max_green_cap:
        # Hard cap reached to prevent cross-traffic starvation
        if best_contender_id != current_phase:
            should_switch = True
            next_phase_id = best_contender_id
            contender_eval = evaluations.get(next_phase_id, curr_eval)
            allocated_green = contender_eval.get("green_extension_seconds", max_green_cap)
            decision_rationale = f"Switching to phase {next_phase_id}: maximum green cap ({max_green_cap}s) reached on phase {current_phase}."
        else:
            allocated_green = curr_eval.get("green_extension_seconds", max_green_cap)
            decision_rationale = f"Retaining phase {current_phase}: no waiting competitors."
    else:
        # Dynamic fuzzy trade-off
        # Switch if competitor has urgent queue/wait and current phase queue has dissipated
        curr_phase_data = phases_dict.get(current_phase, {})
        curr_queue = float(curr_phase_data.get("queue_length", curr_phase_data.get("queue", 0.0)))
        if contender_urgency > curr_urgency * 1.35 and curr_queue < 8.0:
            should_switch = True
            next_phase_id = best_contender_id
            contender_eval = evaluations.get(next_phase_id, curr_eval)
            allocated_green = contender_eval.get("green_extension_seconds", 30.0)
            decision_rationale = f"Switching to phase {next_phase_id}: queue on phase {current_phase} cleared, competitor urgency ({contender_urgency:.1f}) exceeds current ({curr_urgency:.1f})."
        else:
            should_switch = False
            next_phase_id = current_phase
            allocated_green = curr_eval.get("green_extension_seconds", 30.0)
            decision_rationale = f"Extending phase {current_phase}: current traffic demands {allocated_green:.1f}s continuation."

    allocated_green = float(np.clip(allocated_green, 10.0, 60.0))
    chosen_eval = evaluations.get(next_phase_id, curr_eval)

    return {
        "next_phase_id": next_phase_id,
        "allocated_green_seconds": round(allocated_green, 2),
        "should_switch": bool(should_switch),
        "decision_rationale": decision_rationale,
        "urgency_scores": {str(k): round(v, 2) for k, v in urgency_scores.items()},
        "active_rules": chosen_eval.get("active_rules", []),
        "fuzzy_details": chosen_eval
    }
