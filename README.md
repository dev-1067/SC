# Smart Traffic Signal Control System (STCS)
### *An Intelligent Transportation System Powered by Soft Computing & Hybrid AI*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8.svg)](https://opencv.org/)
[![WebSockets](https://img.shields.io/badge/WebSockets-Real--Time-orange.svg)](https://websockets.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Executive Summary & Project Objectives
Urban traffic congestion leads to immense economic losses, excess fuel consumption, elevated greenhouse emissions, and emergency vehicle blockages. Traditional static timer-based signals (e.g., fixed 30s intervals) fail catastrophically under fluctuating traffic density.

The **Smart Traffic Signal Control System (STCS)** dynamically adjusts green-light duration and optimizes intersection phase sequences. It integrates four core Soft Computing paradigms:
* **Artificial Neural Networks (ANN):** Multi-Layer Perceptron (MLP) trained with backpropagation to forecast 10-minute traffic volume and classify surge pressure.
* **Fuzzy Logic Controllers (FLC):** 36-rule Mamdani Fuzzy Inference System with Ruspini-compliant partitions and Centroid (COG) defuzzification under imprecision.
* **Genetic Algorithms (GA):** Real-coded population optimizer tuning triangular and trapezoidal membership function limits, yielding $\ge 25\%$ delay reduction.
* **Adaptive Neuro-Fuzzy Inference Systems (ANFIS):** 5-layer Takagi-Sugeno network utilizing Jang's hybrid learning (LSE forward + Backprop backward) with $R^2 \ge 0.99$.

---

## 2. Syllabus to Project Mapping

| Syllabus Module | Syllabus Core Topics | Application in STCS |
| :--- | :--- | :--- |
| **Module 1: Introduction to Soft Computing** | Soft vs. Hard computing, imprecision tolerance, neural, fuzzy, and hybrid concepts | Managing noisy camera detection errors and ambiguous vehicle wait states without discrete failure. |
| **Module 2: Artificial Neural Networks** | Biological vs. artificial neurons, activation functions, learning rules, adaptation | Sensor fusion layer weighting inputs (vehicle counts, waiting times, flow rates) to classify intersection saturation. |
| **Module 3: Types of Neural Networks** | Linearly vs. non-linearly separable classification, Perceptron convergence, MLP, Backpropagation | PyTorch Multi-Layer Perceptron (6-32-16-2) trained via error backpropagation to forecast traffic density for subsequent 10-minute time horizons. |
| **Module 4: Fuzzy Set Theory** | Crisp vs. Fuzzy sets, Membership Functions (Triangular, Trapezoidal), Fuzzification, Defuzzification (Centroid, CoA) | Fuzzifying crisp vehicle queues ($Q$), waiting times ($W$), and arrival rates ($V$). Defuzzifying via Centroid COG to compute green light extension ($\Delta T_{green}$). |
| **Module 5: Fuzzy Rule Base & FLC** | Fuzzy propositions, rule composition, Mamdani and Takagi-Sugeno inference, FLC architecture | Rule base engine (36 IF-THEN matrices) executing real-time multi-criteria decision making for dynamic phase switching. |
| **Module 6: GA & Hybrid Systems** | Chromosome encoding, crossover, mutation, selection, ANFIS architecture, hybrid learning | GA optimizes boundary parameters of fuzzy membership functions; 5-layer ANFIS tunes Takagi-Sugeno consequent parameters offline. |

---

## 3. High-Level System Architecture

```text
               [ Synthetic / Real Video Stream ]
                               │
                               ▼
        [ Vehicle Detection & Queue Extraction (OpenCV MOG2) ]
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
[ Module 2 & 3: Backprop MLP ]       [ Real-Time Inflow Telemetry ]
(10-Min Horizon Flow Forecast)                    │
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
     [ Module 4 & 5: Mamdani Fuzzy Logic Controller (FLC) ]
          - Fuzzification: Queue Length, Wait Time, Flow Rate
          - Inference: 36 Evaluated IF-THEN Rules
          - Defuzzification: Centroid of Area (COG)
                               ▲
                               │
              [ Module 6: Offline Optimization ]
          - Genetic Algorithm (GA) Membership Tuner
          - 5-Layer TS-Kang ANFIS Engine (Jang Hybrid Learning)
                               │
                               ▼
             [ Phase Switching Decision Engine ]
        (Minimum 10s Green / Maximum 60s Cap / 3s Yellow)
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
[ Microscopic Simulation Engine ]    [ Central Orchestrator & SQLite DB ]
(4-Way Intersection Car Following)                │
                                                  ▼
                                     [ FastAPI REST & WebSocket Server ]
                                                  │
                                                  ▼
                               [ Interactive HTML5 Canvas Cockpit ]
```

---

## 4. Mathematical Formulations

### 4.1 Traffic Flow Prediction (Backprop MLP)
* **Input Feature Vector:**
  $$X = [q_t, q_{t-1}, v_t, w_t, \text{hour\_of\_day}, \text{is\_rush\_hour}]^T \in \mathbb{R}^6$$
* **Layer Propagation:**
  $$z^{[1]} = W^{[1]}X + b^{[1]}, \quad a^{[1]} = \text{ReLU}(z^{[1]})$$
  $$z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}, \quad a^{[2]} = \text{ReLU}(z^{[2]})$$
  $$\hat{y} = W^{[3]}a^{[2]} + b^{[3]} = [\hat{q}_{t+10}, \hat{s}_{\text{factor}}]^T$$
* **Loss Function:**
  $$J(W, b) = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2 + \lambda \|W\|_2^2$$

### 4.2 Mamdani Fuzzy Logic Controller (Centroid Defuzzification)
* **Membership Functions:** Ruspini partition compliant ($\sum \mu_i(x) = 1.0$).
* **Fuzzy Inference:** Min-Max compositional rule of inference.
* **Centroid Defuzzification (Center of Gravity):**
  $$\Delta T_{\text{green}} = \frac{\int_{T_{\min}}^{T_{\max}} t \cdot \mu_{\text{aggregated}}(t) \, dt}{\int_{T_{\min}}^{T_{\max}} \mu_{\text{aggregated}}(t) \, dt}$$

### 4.3 Genetic Algorithm MF Parameter Optimization
* **Chromosome:** Real-coded vector $\vec{\theta} \in \mathbb{R}^{37}$ representing triangular peaks and trapezoidal boundary bounds.
* **Objective Fitness Function:**
  $$\min f(\vec{\theta}) = \sum_{t=1}^{T_{\text{eval}}} \sum_{\text{lanes } l} \left( Q_l(t) \cdot W_l(t) \right) + \text{Penalty}_{\text{monotonicity}}$$
* **Operators:** Simulated Binary Crossover (SBX, $\eta_c = 2.0$), Polynomial Mutation ($\eta_m = 20.0$), Tournament Selection ($k=3$).

### 4.4 5-Layer ANFIS Architecture (Takagi-Sugeno)
* **Layer 1 (Fuzzification):** Node outputs are fuzzy membership grades using Gaussian / Generalized Bell MFs.
* **Layer 2 (Rule Firing):** T-norm product operator: $w_i = \mu_{A_i}(x) \times \mu_{B_i}(y) \times \mu_{C_i}(z)$.
* **Layer 3 (Normalization):** $\bar{w}_i = \frac{w_i}{\sum_{j=1}^{R} w_j}$.
* **Layer 4 (Consequents):** First-order polynomial: $f_i = p_i x + q_i y + r_i z + s_i$, output is $\bar{w}_i f_i$.
* **Layer 5 (Overall Sum):** Final crisp output: $y = \sum_{i=1}^{R} \bar{w}_i f_i$.

---

## 5. Web Cockpit & Interactive Dashboard Features

The full-stack cockpit running at `http://localhost:8000` provides:
1. **HTML5 60 FPS Canvas:** Real-time 2D animated intersection rendering with car-following kinematics, stop lines, queuing lanes, and dynamic signal states (Red, Yellow, Green).
2. **Control Mode Switcher:** Seamless cycling between `ADAPTIVE` (Soft Computing FLC + ANN), `STATIC_FIXED` (30s baseline), and `MANUAL` (operator control).
3. **Speed Multipliers:** 1x, 2x, and 5x simulation clock acceleration.
4. **Pause / Resume / Reset:** Instant clock freezing and state reset.
5. **Emergency Vehicle Preemption:** Priority corridor override (North, South, East, West) forcing immediate clearance for first responders with audio-visual alerts.
6. **Dynamic Inflow Sliders:** Interactive vehicle arrival sliders (0.0 to 15.0 veh/min) for stress-testing bottleneck scenarios.
7. **Automated Comparative Benchmark:** 150-second surge test evaluating Static vs. Adaptive performance across Delay Reduction ($\ge 25\%$), Throughput Gain, and Queue Spillback.
8. **Soft Computing Architecture Inspector:** Deep structural modal visualizing rules, MFs, and neural activations.
9. **SQLite Audit Trail & Export:** Persistent historical logging with one-click JSON export.

---

## 6. REST API & WebSocket Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves interactive HTML5 Canvas dashboard |
| `WS` | `/ws/traffic-stream` | High-frequency 10 Hz real-time telemetry broadcast |
| `GET` | `/api/status` | Current system telemetry snapshot |
| `POST` | `/api/control/mode` | Sets control mode (`ADAPTIVE`, `STATIC_FIXED`, `MANUAL`) |
| `POST` | `/api/control/switch` | Forces immediate manual phase transition with 3s yellow safety |
| `POST` | `/api/control/emergency` | Triggers emergency vehicle priority preemption hold |
| `POST` | `/api/control/emergency/clear` | Clears emergency override and returns to autonomous control |
| `POST` | `/api/control/demand` | Updates arrival inflow rates per approach (veh/min) |
| `POST` | `/api/control/speed` | Sets simulation clock multiplier (0.5x to 5.0x) |
| `POST` | `/api/control/pause` | Pauses simulation clock |
| `POST` | `/api/control/resume` | Resumes simulation clock |
| `POST` | `/api/control/reset` | Resets simulation environment to clean initial state |
| `POST` | `/api/benchmark/run` | Executes head-to-head comparative surge benchmark |
| `GET` | `/api/soft-computing/models`| Returns structural details and parameters of AI models |
| `GET` | `/api/metrics/history` | Historical performance logs for telemetry charts |
| `GET` | `/api/switches/history` | Signal switching audit trail with decision rationales |
| `GET` | `/api/benchmarks/history` | Historical comparative benchmark experiment results |

---

## 7. Project Structure

```text
.
├── main.py                     # FastAPI REST API & WebSocket server
├── orchestrator.py             # Central simulation event loop & WebSocket broadcaster
├── fuzzy_engine.py             # Mamdani FLC, membership functions & defuzzification
├── ga_optimizer.py             # Genetic Algorithm optimizer for fuzzy MF parameters
├── anfis_tuner.py              # 5-Layer Takagi-Sugeno ANFIS with hybrid learning
├── ann_predictor.py            # PyTorch MLP backpropagation traffic flow predictor
├── vision_detector.py          # OpenCV MOG2 queue detection & vehicle tracking
├── simulation_env.py           # 4-way microscopic crossroad simulation with car physics
├── database.py                 # SQLite ORM storage for metrics, switches & benchmarks
├── optimized_fuzzy_params.json # GA-optimized fuzzy membership parameters
├── anfis_traffic_model.pt      # Pretrained PyTorch weights for ANFIS model
├── ann_traffic_model.pt        # Pretrained PyTorch weights for MLP predictor
├── ann_metadata.json           # Normalization statistics for ANN inputs/outputs
├── static/
│   ├── index.html              # Cockpit UI structure, Lucide icons & modals
│   ├── style.css               # Modern glassmorphic dark-mode CSS design system
│   └── dashboard.js            # 60 FPS HTML5 canvas renderer & WebSocket client
├── demo_person1.py             # Interactive demonstration of Person 1 deliverables
├── demo_person2.py             # Interactive demonstration of Person 2 deliverables
├── test_person1.py             # Pytest suite for Fuzzy Logic, GA & ANFIS
├── test_person2.py             # Pytest suite for ANN Predictor, Vision & Simulation
├── test_person3.py             # Pytest suite for FastAPI, WebSockets & Database
├── test_cockpit_e2e.py         # End-to-end API & WebSocket verification suite
├── requirements.txt            # Python dependencies
└── README.md                   # Complete architectural & operational guide
```

---

## 8. Installation & Quickstart

### Prerequisites
* Python 3.10 or higher
* Git

### 1. Clone the Repository
```bash
git clone https://github.com/dev-1067/SC.git
cd SC
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Server
```bash
python main.py
```
Open your browser and navigate to:
**`http://localhost:8000`**

### 4. Run Automated Test Suites
```bash
# Run all 47 unit & integration tests
pytest -v

# Run the complete end-to-end REST & WebSocket verification
python test_cockpit_e2e.py
```

### 5. Run Interactive CLI Demos
```bash
# Person 1: FLC, Genetic Algorithm & ANFIS demonstration
python demo_person1.py

# Person 2: ANN Predictor, OpenCV Detection & 150s Benchmark
python demo_person2.py
```

---

## 9. Verification & Performance Results
* **47 / 47 Automated Pytest Tests Passing** (100% pass rate).
* **Average Delay Reduction:** $+29.6\%$ to $+31.0\%$ over standard fixed-time signals (exceeds $\ge 25\%$ target).
* **ANN Prediction Fidelity:** $R^2 = 0.9940$ on 10-minute traffic volume forecasting.
* **ANFIS Convergence:** $R^2 = 0.9938$ with $96.3\%$ loss reduction in 15 epochs.
* **Vision Processing Throughput:** $\sim 45$ FPS real-time tracking with background subtraction.
