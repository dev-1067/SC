# Smart Traffic Signal Control System (STCS)
### *An Intelligent Transportation System Powered by Soft Computing & Hybrid AI*

---

## 1. Executive Summary & Project Objectives
Urban traffic congestion leads to substantial economic losses, excess fuel consumption, elevated greenhouse emissions, and emergency vehicle blockages. Static timer-based signals fail under fluctuating traffic density.

The **Smart Traffic Signal Control System (STCS)** dynamically adjusts green-light duration and optimizes intersection phase sequences. It solves traffic scheduling via core Soft Computing paradigms:
* **Artificial Neural Networks (ANN):** Flow forecasting and lane load classification.
* **Fuzzy Logic Controllers (FLC):** Rule-based phase timing under uncertainty and vagueness.
* **Genetic Algorithms (GA):** Search-space optimization for fuzzy membership function bounds.
* **Adaptive Neuro-Fuzzy Inference Systems (ANFIS):** Hybrid learning for self-tuning control parameters.

---

## 2. Complete Syllabus to Project Mapping

| Syllabus Module | Syllabus Core Topics | Application in STCS |
| :--- | :--- | :--- |
| **Module 1: Introduction to Soft Computing** | Soft vs. Hard computing, imprecision tolerance, neural, fuzzy, and hybrid concepts | Managing noisy, real-world camera detection errors and ambiguous vehicle wait states without discrete failure. |
| **Module 2: Artificial Neural Networks** | Biological vs. artificial neurons, activation functions, learning rules, adaptation | Sensor fusion layer: weighting inputs (vehicle counts, waiting times, weather index) to classify intersection saturation. |
| **Module 3: Types of Neural Networks** | Linearly vs. non-linearly separable classification, Perceptron convergence, MLP, Backpropagation, ART/SOM | Multi-Layer Perceptron (MLP) trained with error backpropagation to forecast traffic density for subsequent 5–15 minute time horizons. |
| **Module 4: Fuzzy Set Theory** | Crisp vs. Fuzzy sets, Membership Functions (Triangular, Trapezoidal), Fuzzification, Defuzzification (Centroid, CoA, Lambda-cuts) | Fuzzifying crisp vehicle counts ($Q$), wait times ($W$), and vehicle arrival rates ($\Delta V$). Defuzzifying output to determine crisp green light extension ($\Delta T_{green}$). |
| **Module 5: Fuzzy Rule Base & FLC** | Fuzzy propositions, rule composition, Mamdani and Takagi-Sugeno inference, FLC architecture | Rule base engine (IF-THEN matrices) executing real-time multi-criteria decision making for dynamic phase switching. |
| **Module 6: GA & Hybrid Systems** | Chromosome encoding, crossover, mutation, selection, ANFIS architecture, hybrid learning | GA optimizes boundary parameters of fuzzy membership functions; ANFIS automatically tunes Takagi-Sugeno rule weights offline. |

---

## 3. High-Level System Architecture

```text
[ Video Stream / Camera Feed ]
             │
             ▼
[ Vehicle Detection & Queue Extraction (YOLOv8 / OpenCV) ]
             │
             ├──► [ Module 2 & 3: Backprop ANN Flow Predictor ]
             │           │
             │           ▼ (Predicted Surge Factor)
             ▼
[ Module 4 & 5: Fuzzy Logic Controller (Mamdani FIS Engine) ]
             │  - Fuzzification: Inputs (Queue Length, Wait Time, Flow)
             │  - Inference: 27 Evaluated IF-THEN Rules
             │  - Defuzzification: Centroid Method (COG)
             │
             ▲
[ Module 6: Offline Optimization (GA & ANFIS) ]
             │  - GA Tunes Membership Function Limits
             │  - ANFIS Adapts Consequent Parameters
             │
             ▼
[ Phase Switching Decision Engine & Green Allocation ]
             │
             ├──► [ Simulation Layer: SUMO (TraCI API) / Pygame ]
             └──► [ Real-Time Dashboard: FastAPI + WebSockets + React ]
```

---

## 4. End-to-End Tech Stack

### A. Soft Computing & AI Engine
* **Language:** Python 3.10+
* **Fuzzy Logic:** `scikit-fuzzy`, custom Python FIS implementation
* **Neural Network & ANFIS:** PyTorch or TensorFlow (`keras.models.Sequential`)
* **Genetic Algorithm:** `DEAP` or vectorized NumPy genetic operators
* **Vision / Sensing:** OpenCV (`cv2`), YOLOv8 (`ultralytics`)

### B. Simulation & Control Execution
* **Simulator:** SUMO (*Simulation of Urban MObility*) using `traci` Python library, or custom 4-way intersection built with `pygame`
* **Orchestrator:** Python event loop synchronizing detector reads with phase assignments

### C. Backend & API Services
* **Framework:** FastAPI
* **Real-time Comms:** Native WebSockets (`/ws/traffic-stream`)
* **Background Tasks:** Celery + Redis (for periodic offline GA/ANFIS parameter updates)

### D. Frontend & Dashboard
* **Framework:** React.js (Vite) + Tailwind CSS + Lucide Icons
* **Visualization:** HTML5 Canvas (intersection graphics) + Recharts (queue length, delay, throughput charts)

### E. Database & Storage
* **Telemetry Store:** PostgreSQL / TimescaleDB (for queue logs and signal change audit trails)
* **In-Memory Cache:** Redis (for real-time active light state and vehicle counts)

---

## 5. Mathematical Formulations & Soft Computing Details

### 5.1 Module 3: Traffic Flow Prediction (Backprop MLP)
* **Input Vector:**
  $$X = [q_t, q_{t-1}, v_t, w_t, \text{hour\_of\_day}, \text{is\_rush\_hour}]^T$$
* **Layer Propagation:**
  $$z^{[1]} = W^{[1]}X + b^{[1]}, \quad a^{[1]} = \text{ReLU}(z^{[1]})$$
  $$\hat{y} = W^{[2]}a^{[1]} + b^{[2]}$$
* **Loss Function:**
  $$J(W, b) = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

### 5.2 Module 4 & 5: Fuzzy Logic Controller (FLC)
* **Membership Functions:** Triangular ($\text{trimf}$) and Trapezoidal ($\text{trapmf}$) curves over:
  * Queue Length ($Q$): `{Short, Medium, Long, Very Long}` on $[0, 50]$ vehicles
  * Waiting Time ($W$): `{Low, Normal, Critical}` on $[0, 180]$ seconds
  * Arrival Rate ($\Delta V$): `{Low, Medium, High}` on $[0, 10]$ vehicles/min
* **Defuzzification (Centroid Method):**
  $$z^* = \frac{\int z \cdot \mu_{\text{output}}(z) \, dz}{\int \mu_{\text{output}}(z) \, dz}$$
  Where $z^*$ represents the crisp green light time extension $\Delta T \in [10, 60]$ seconds.

### 5.3 Module 6: Genetic Algorithm Optimization
* **Chromosome Representation:** Real-coded vector encoding fuzzy set boundaries:
  $$C = [a_1, b_1, c_1, a_2, b_2, c_2, \dots, a_k, b_k, c_k]$$
* **Fitness Function:**
  $$\text{Minimize } \mathcal{F} = \sum_{t=1}^{T} \left( \sum_{i=1}^{\text{lanes}} \text{Queue}_i(t) \cdot \text{WaitTime}_i(t) \right)$$
* **Operators:** Roulette-wheel / Tournament selection, simulated binary crossover (SBX), and polynomial mutation.

---

## 6. Work Breakdown: 3-Person Division of Labor

The project is broken down into three distinct, non-overlapping modules with explicit integration contracts. When all three developers complete their sections, the system achieves 100% completion.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        PERSON 1: SOFT COMPUTING CORE                   │
│   • Module 4 & 5: FLC Engine (Fuzzification, Rules, Defuzzification)   │
│   • Module 6: GA Optimizer & ANFIS Self-Tuner                          │
│   • Decision Engine Library (Exposes: `calculate_green_extension()`)   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Shared Python Module
┌──────────────────────────────────┴─────────────────────────────────────┐
│                   PERSON 2: DATA, SENSING & NEURAL NETS                │
│   • Module 2 & 3: Backpropagation MLP (Congestion Predictor)           │
│   • Vehicle Detection & Queue Extraction (YOLOv8 / OpenCV)             │
│   • Traffic Simulation Environment (SUMO / Pygame Engine)              │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Telemetry & State Stream
┌──────────────────────────────────┴─────────────────────────────────────┐
│                    PERSON 3: BACKEND, API & FRONTEND                   │
│   • FastAPI Server, WebSockets, Redis Cache & Database Models         │
│   • Real-Time Control Loop Orchestration (Hardware / Sim Actuation)    │
│   • React Web Dashboard: 2D Visualizer, Signal Countdown, Analytics   │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Person 1: Soft Computing, Fuzzy Logic & Optimization Engineer
**Core Focus:** Modules 4, 5, and 6 (Syllabus Core)

* **Key Deliverables:**
  1. **Fuzzy Inference System (`fuzzy_engine.py`):**
     * Define linguistic variables for input membership functions (Triangular & Trapezoidal) for Queue Length, Waiting Time, and Approaching Vehicles.
     * Define output variable: Green Signal Time Extension ($10\text{s} - 60\text{s}$).
     * Implement Mamdani inference engine with a minimum of 27 IF-THEN rules.
     * Code the Defuzzification stage using the Centroid / Center of Gravity (COG) method.
  2. **Genetic Algorithm Optimizer (`ga_optimizer.py`):**
     * Formulate real-coded chromosomes encoding the vertex coordinates of the fuzzy membership functions.
     * Implement crossover, mutation, and selection operators.
     * Create an automated tuning script that optimizes fuzzy bounds to minimize cumulative intersection delay.
  3. **ANFIS Module (`anfis_tuner.py`):**
     * Construct 5-layer ANFIS model for Sugeno-type inference to adapt rule consequents using gradient descent and least-squares estimation.
  4. **Integration Hook:**
     * Provide a unified Python function:
       ```python
       def get_optimal_phase(lane_metrics: dict) -> dict:
           """Returns next phase ID and allocated green light duration in seconds."""
       ```

---

### Person 2: Sensing, Neural Networks & Traffic Simulation Engineer
**Core Focus:** Modules 1, 2, 3 + Simulation Environment

* **Key Deliverables:**
  1. **ANN Traffic Flow Predictor (`ann_predictor.py`):**
     * Design a Multi-Layer Perceptron (MLP) using PyTorch / Keras with backpropagation learning.
     * Train the model on historical traffic pattern data (time of day, day of week, upstream flow) to predict expected queue load 10 minutes ahead.
     * Export lightweight inference pipeline to feed predicted load into the controller.
  2. **Vehicle Counting & Queue Estimator (`vision_detector.py`):**
     * Integrate YOLOv8 or OpenCV background subtraction on simulated junction camera feeds.
     * Generate crisp lane counts: current stationary vehicles and incoming vehicle speed.
  3. **Traffic Simulation Environment (`simulation_env.py`):**
     * Configure a 4-way crossroad junction in **SUMO** (using `.net.xml`, `.rou.xml`) or build an interactive simulation in **Pygame**.
     * Expose vehicle positions, lane queues, and waiting timers to the system.
     * Connect to the control actuators using `traci` (or Pygame signal state variables) to dynamically toggle RED, YELLOW, and GREEN lights.

---

### Person 3: Full-Stack Infrastructure, Orchestration & Dashboard Engineer
**Core Focus:** System Integration, Backend, Database, Real-Time UI

* **Key Deliverables:**
  1. **Backend Orchestrator & API (`main.py`):**
     * Build FastAPI application serving REST endpoints for configuration and manual overrides.
     * Implement high-frequency WebSocket endpoint (`/ws/traffic-live`) broadcasting live vehicle queues, active phase, and countdown timer.
     * Write the central loop that retrieves vehicle counts from **Person 2**, sends them to **Person 1's** Soft Computing model, and writes the resulting signal decisions back to the simulation.
  2. **Database & Telemetry Storage (`database.py`):**
     * Set up PostgreSQL/SQLite with tables for metric logs, signal switching history, and system performance benchmarks.
  3. **Monitoring & Control Dashboard (React.js + Tailwind CSS):**
     * **Interactive 2D Junction View:** Visual representation of the 4-way intersection with real-time changing signal lights (Red/Yellow/Green) and vehicle counts.
     * **Live Telemetry Charts:** Line charts tracking Average Delay, Waiting Time per Lane, and Throughput Over Time.
     * **Soft Computing Inspection Panel:** Live graphic display showing activated fuzzy rules and membership function values for the current cycle.
     * **Control Mode Switch:** Toggle between "Static Fixed-Time" and "Adaptive Soft Computing" modes for comparative live demonstration.

---

## 7. Joint Integration & Verification Checklist

When these milestones are reached, the project is **100% complete and verified**:

- [ ] **Milestone 1 (Person 1 + Person 2):** Person 2 streams synthetic or simulated queue data into Person 1's FLC function; FLC outputs valid green times (10s–60s) without crashes.
- [ ] **Milestone 2 (Person 2 + Person 3):** Person 3's backend connects to Person 2's SUMO/Pygame simulation via WebSocket/socket loop, correctly reading queue counts and toggling signal phases.
- [ ] **Milestone 3 (Person 1 + Person 3):** Dashboard correctly displays activated fuzzy membership functions, inference rules, and neural network predictions in real time.
- [ ] **Final Project Benchmark (All 3):** Run a 1-hour simulation benchmark comparing standard fixed-time control (e.g., 30s static) vs. STCS. Produce a comparative report demonstrating:
  * $\ge 25\%$ reduction in average vehicle waiting time.
  * $\ge 20\%$ reduction in queue spillback during peak surge hours.
  * Complete compliance with all 6 modules of the Soft Computing syllabus.