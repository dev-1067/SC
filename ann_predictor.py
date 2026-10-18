"""
Smart Traffic Signal Control System (STCS)
Person 2 Deliverable: ANN Traffic Flow Predictor (ann_predictor.py)
Covers Syllabus Modules 2 & 3:
- Multi-Layer Perceptron (MLP) with Backpropagation Learning
- Non-linear separation via ReLU activation
- Input Vector X = [q_t, q_{t-1}, v_t, w_t, hour_of_day, is_rush_hour]^T
- Forward Propagation: z^[1] = W^[1]X + b^[1], a^[1] = ReLU(z^[1]), y_hat = W^[2]a^[1] + b^[2]
- Loss Function: MSE J(W, b) = (1/N) * sum((y_i - y_hat_i)^2)
- 10-Minute-Ahead Queue & Surge Forecasting Pipeline
"""

import json
import os
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ==============================================================================
# 1. MULTI-LAYER PERCEPTRON (MLP) ARCHITECTURE (Modules 2 & 3)
# ==============================================================================

class TrafficFlowMLP(nn.Module):
    """
    Multi-Layer Perceptron for Traffic Flow & Queue Forecasting.
    Mathematical formulation:
        Input: X = [q_t, q_{t-1}, v_t, w_t, hour_of_day, is_rush_hour]^T (dim=6)
        Hidden Layer 1: z^[1] = W^[1]X + b^[1], a^[1] = ReLU(z^[1])
        Hidden Layer 2: z^[2] = W^[2]a^[1] + b^[2], a^[2] = ReLU(z^[2])
        Output Layer: y_hat = W^[3]a^[2] + b^[3] (dim=2: [predicted_queue_10min, surge_factor])
    """

    def __init__(self, input_dim: int = 6, hidden_dim1: int = 32, hidden_dim2: int = 16, output_dim: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim

        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        # Layer 2
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        # Layer 3 (Output: [predicted_queue, surge_factor])
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

        # Initialize weights using He/Kaiming initialization (Module 2 learning rule adaptation)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation through all layers."""
        z1 = self.fc1(x)
        a1 = self.relu1(z1)
        z2 = self.fc2(a1)
        a2 = self.relu2(z2)
        out = self.fc3(a2)
        return out

    def get_layer_activations(self, x: torch.Tensor) -> Dict[str, Any]:
        """Introspection helper exposing neuron activations for Person 3 Dashboard."""
        with torch.no_grad():
            z1 = self.fc1(x)
            a1 = self.relu1(z1)
            z2 = self.fc2(a1)
            a2 = self.relu2(z2)
            out = self.fc3(a2)
            return {
                "layer1_z": z1.cpu().numpy().tolist()[0],
                "layer1_a": a1.cpu().numpy().tolist()[0],
                "layer2_z": z2.cpu().numpy().tolist()[0],
                "layer2_a": a2.cpu().numpy().tolist()[0],
                "output": out.cpu().numpy().tolist()[0]
            }


# ==============================================================================
# 2. DATASET GENERATOR (Simulating Historical Traffic Telemetry)
# ==============================================================================

class TrafficDatasetGenerator:
    """
    Generates realistic historical diurnal traffic dataset with morning/evening rush hours,
    temporal autocorrelation, and weather/upstream variations.
    """

    @staticmethod
    def generate_data(n_samples: int = 2400, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Creates feature matrix X: [N, 6] and target matrix Y: [N, 2]
        Features:
            0: q_t (current queue length, [0, 50])
            1: q_t_prev (previous queue length, [0, 50])
            2: v_t (arrival rate, [0, 10] veh/min)
            3: w_t (average wait time, [0, 180] s)
            4: hour_of_day ([0, 24])
            5: is_rush_hour (1.0 if (7 <= h <= 10 or 16 <= h <= 19) else 0.0)
        Targets:
            0: q_t_plus_10 (queue length 10 mins ahead)
            1: surge_factor ([0.0, 1.0] intensity index)
        """
        np.random.seed(seed)

        hours = np.random.uniform(0.0, 24.0, size=(n_samples, 1))
        is_rush = np.zeros_like(hours)
        for i, h in enumerate(hours):
            val = h[0]
            if (7.0 <= val <= 10.0) or (16.0 <= val <= 19.5):
                is_rush[i] = 1.0

        # Base arrival rate depends on time of day (diurnal curve)
        diurnal_flow = 2.0 + 5.5 * np.exp(-0.5 * ((hours - 8.5) / 1.8) ** 2) + \
                       6.0 * np.exp(-0.5 * ((hours - 17.5) / 2.0) ** 2)
        v_t = np.clip(diurnal_flow + np.random.normal(0.0, 0.8, size=(n_samples, 1)), 0.5, 10.0)

        # Autoregressive queue relationship: q_t depends on flow and past state
        q_t_prev = np.clip(v_t * 3.5 + np.random.normal(0.0, 3.0, size=(n_samples, 1)), 0.0, 50.0)
        q_t = np.clip(0.6 * q_t_prev + 0.4 * (v_t * 4.0) + np.random.normal(0.0, 2.0, size=(n_samples, 1)), 0.0, 50.0)

        # Wait time correlates with queue length
        w_t = np.clip(q_t * 3.2 + np.random.normal(0.0, 8.0, size=(n_samples, 1)), 0.0, 180.0)

        # Target: expected queue 10 minutes ahead
        # If rush hour and flow is high, queue builds rapidly
        growth_rate = np.where(is_rush == 1.0, 1.25, 0.85)
        q_target = np.clip(q_t * growth_rate + (v_t * 1.5) - 3.0 + np.random.normal(0.0, 1.5, size=(n_samples, 1)), 0.0, 50.0)

        # Surge factor: normalized surge pressure
        surge_target = np.clip((q_target / 50.0) * 0.6 + (v_t / 10.0) * 0.4, 0.0, 1.0)

        X_raw = np.hstack([q_t, q_t_prev, v_t, w_t, hours, is_rush])
        Y_raw = np.hstack([q_target, surge_target])

        # Normalization statistics
        mean_X = np.mean(X_raw, axis=0)
        std_X = np.std(X_raw, axis=0) + 1e-8
        stats = {
            "mean_X": mean_X.tolist(),
            "std_X": std_X.tolist(),
            "feature_names": ["q_t", "q_t_prev", "v_t", "w_t", "hour_of_day", "is_rush_hour"],
            "target_names": ["q_t_plus_10", "surge_factor"]
        }

        # Normalize features
        X_norm = (X_raw - mean_X) / std_X

        return torch.tensor(X_norm, dtype=torch.float32), torch.tensor(Y_raw, dtype=torch.float32), stats


# ==============================================================================
# 3. BACKPROPAGATION TRAINER & FLOW PREDICTOR
# ==============================================================================

class ANNTrafficPredictor:
    """
    High-Level Traffic Flow Prediction Engine.
    Handles training via Backpropagation with MSE loss, parameter persistence,
    and real-time inference for Person 1 (FLC) and Person 3 (API).
    """

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrafficFlowMLP().to(self.device)
        self.stats: Optional[Dict[str, Any]] = None

        default_model_path = os.path.join(os.path.dirname(__file__), "ann_traffic_model.pt")
        target_path = model_path or default_model_path
        if os.path.exists(target_path):
            try:
                self.load_model(target_path)
            except Exception:
                pass

    def train_backprop(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        stats: Dict[str, Any],
        epochs: int = 40,
        batch_size: int = 64,
        lr: float = 0.005,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Trains the MLP using error backpropagation and Adam optimizer.
        """
        self.stats = stats
        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        n_samples = X_train.shape[0]
        history = []

        with torch.no_grad():
            init_loss = criterion(self.model(X_train), Y_train).item()

        if verbose:
            print("=" * 70)
            print("Training MLP Traffic Flow Predictor with Error Backpropagation")
            print(f"Samples: {n_samples} | Features: 6 | Targets: 2 | Initial MSE: {init_loss:.4f}")
            print("=" * 70)

        for epoch in range(1, epochs + 1):
            self.model.train()
            permutation = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                idx = permutation[i:i + batch_size]
                bx, by = X_train[idx], Y_train[idx]

                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            history.append({"epoch": epoch, "mse_loss": round(avg_loss, 4)})

            if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
                print(f"Epoch {epoch:02d}/{epochs:02d} | Training MSE: {avg_loss:8.4f} | "
                      f"Loss Reduction: {((init_loss - avg_loss)/init_loss)*100:5.2f}%")

        # Evaluate performance
        self.model.eval()
        with torch.no_grad():
            final_pred = self.model(X_train).cpu().numpy()
            y_true = Y_train.cpu().numpy()
            mse = float(np.mean((final_pred - y_true) ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(final_pred - y_true)))
            r2 = float(1.0 - (np.sum((y_true - final_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8)))

        if verbose:
            print("=" * 70)
            print("Backpropagation Training Complete!")
            print(f"Final Metrics -> MSE: {mse:.4f} | RMSE: {rmse:.3f} | MAE: {mae:.3f} | R^2: {r2:.4f}")
            print("=" * 70)

        return {
            "initial_loss": init_loss,
            "final_mse": round(mse, 4),
            "final_rmse": round(rmse, 4),
            "final_mae": round(mae, 4),
            "r2_score": round(r2, 4),
            "history": history
        }

    def predict_flow(
        self,
        q_t: float,
        q_t_prev: float,
        v_t: float,
        w_t: float,
        hour_of_day: float = 12.0
    ) -> Dict[str, Any]:
        """
        Real-Time Lightweight Inference Pipeline.
        Predicts expected queue 10 minutes ahead and surge factor.
        """
        # Defensive sanitization with NaN/Inf protection
        def _sanitize(val, low: float, high: float, default: float) -> float:
            try:
                f = float(val)
                if math.isnan(f) or math.isinf(f):
                    return default
                return float(np.clip(f, low, high))
            except (ValueError, TypeError):
                return default

        q = _sanitize(q_t, 0.0, 50.0, 0.0)
        q_prev = _sanitize(q_t_prev, 0.0, 50.0, q)
        v = _sanitize(v_t, 0.0, 10.0, 0.0)
        w = _sanitize(w_t, 0.0, 180.0, 0.0)
        h = _sanitize(hour_of_day, 0.0, 24.0, 12.0) % 24.0

        is_rush = 1.0 if (7.0 <= h <= 10.0 or 16.0 <= h <= 19.5) else 0.0
        raw_feat = np.array([q, q_prev, v, w, h, is_rush], dtype=np.float32)

        # Normalize if statistics exist
        if self.stats and "mean_X" in self.stats:
            mean = np.array(self.stats["mean_X"], dtype=np.float32)
            std = np.array(self.stats["std_X"], dtype=np.float32)
            norm_feat = (raw_feat - mean) / std
        else:
            norm_feat = raw_feat

        x_tensor = torch.tensor(norm_feat.reshape(1, -1), dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model(x_tensor).cpu().numpy()[0]
            activations = self.model.get_layer_activations(x_tensor)

        pred_queue = float(np.clip(out[0], 0.0, 50.0))
        surge_factor = float(np.clip(out[1], 0.0, 1.0))

        # Classification flag
        is_surge_imminent = bool(surge_factor > 0.60 or (pred_queue > q * 1.3 and pred_queue > 20.0))

        return {
            "predicted_queue_10min": round(pred_queue, 2),
            "current_queue": round(q, 2),
            "predicted_queue_delta": round(pred_queue - q, 2),
            "surge_factor": round(surge_factor, 4),
            "is_surge_imminent": is_surge_imminent,
            "inputs": {
                "q_t": q, "q_t_prev": q_prev, "v_t": v, "w_t": w,
                "hour_of_day": h, "is_rush_hour": bool(is_rush)
            },
            "layer_activations": activations
        }

    def save_model(self, file_path: str = "ann_traffic_model.pt", stats_path: str = "ann_metadata.json") -> str:
        """Saves PyTorch weights and normalization metadata."""
        torch.save(self.model.state_dict(), file_path)
        if self.stats:
            with open(stats_path, "w") as f:
                json.dump(self.stats, f, indent=2)
        return file_path

    def load_model(self, file_path: str = "ann_traffic_model.pt", stats_path: str = "ann_metadata.json"):
        """Loads weights and normalization metadata."""
        checkpoint = torch.load(file_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                self.stats = json.load(f)


# Singleton instance
_GLOBAL_ANN_PREDICTOR: Optional[ANNTrafficPredictor] = None


def get_ann_predictor() -> ANNTrafficPredictor:
    """Returns singleton instance of ANNTrafficPredictor."""
    global _GLOBAL_ANN_PREDICTOR
    if _GLOBAL_ANN_PREDICTOR is None:
        _GLOBAL_ANN_PREDICTOR = ANNTrafficPredictor()
        # Train baseline if not already trained
        if _GLOBAL_ANN_PREDICTOR.stats is None:
            X, Y, stats = TrafficDatasetGenerator.generate_data(n_samples=1500)
            _GLOBAL_ANN_PREDICTOR.train_backprop(X, Y, stats, epochs=25, verbose=False)
            _GLOBAL_ANN_PREDICTOR.save_model()
    return _GLOBAL_ANN_PREDICTOR


if __name__ == "__main__":
    predictor = ANNTrafficPredictor()
    X, Y, stats = TrafficDatasetGenerator.generate_data(n_samples=2000)
    res = predictor.train_backprop(X, Y, stats, epochs=30, verbose=True)
    predictor.save_model()

    demo = predictor.predict_flow(q_t=22.0, q_t_prev=16.0, v_t=7.5, w_t=65.0, hour_of_day=8.5)
    print("\nSample Flow Forecast (8:30 AM Rush Hour):")
    print(f"  Current Queue:    {demo['current_queue']} vehicles")
    print(f"  Predicted Queue:  {demo['predicted_queue_10min']} vehicles (+{demo['predicted_queue_delta']})")
    print(f"  Surge Intensity:  {demo['surge_factor'] * 100:.1f}% (Surge Imminent: {demo['is_surge_imminent']})")
