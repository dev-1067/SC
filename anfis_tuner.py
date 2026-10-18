"""
Smart Traffic Signal Control System (STCS)
Person 1 Deliverable: Adaptive Neuro-Fuzzy Inference System (ANFIS) (anfis_tuner.py)
Covers Syllabus Modules 2 & 6:
- 5-Layer ANFIS Architecture (Takagi-Sugeno-Kang First-Order Inference)
  * Layer 1: Fuzzification Layer (Gaussian Membership Neurons with learnable c, sigma)
  * Layer 2: Rule Firing Strength Layer (Product T-norm)
  * Layer 3: Normalization Layer (Relative rule activations)
  * Layer 4: Consequent Polynomial Layer (First-order Sugeno: p1*x1 + p2*x2 + p3*x3 + r)
  * Layer 5: Defuzzification / Summation Output Layer
- Hybrid Learning Algorithm:
  * Forward Pass: Least-Squares Estimation (LSE) for consequent parameters
  * Backward Pass: Error Backpropagation / Gradient Descent for premise parameters
- Offline Self-Tuning and Model Serialization
"""

import json
import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ==============================================================================
# 1. 5-LAYER ANFIS PYTORCH ARCHITECTURE (Module 6)
# ==============================================================================

class ANFISNet(nn.Module):
    """
    First-Order Takagi-Sugeno ANFIS Network for 3-Input Traffic Signal Extension.
    Inputs:
        x1: Queue Length (vehicles, [0, 50])
        x2: Waiting Time (seconds, [0, 180])
        x3: Arrival Rate (vehicles/min, [0, 10])
    Output:
        y: Green Signal Extension (seconds, [10, 60])
    """

    def __init__(self, num_mfs_per_input: Tuple[int, int, int] = (3, 3, 3)):
        super().__init__()
        self.num_mfs = num_mfs_per_input
        self.num_rules = int(np.prod(num_mfs_per_input))  # 3 x 3 x 3 = 27 rules

        # ----------------------------------------------------------------------
        # LAYER 1: Premise Parameters (Centers c and Spreads sigma for Gaussian MFs)
        # mu(x) = exp(-0.5 * ((x - c) / sigma)^2)
        # ----------------------------------------------------------------------
        # Initialize centers evenly spaced over each input's domain
        q_centers = torch.linspace(5.0, 45.0, self.num_mfs[0])
        w_centers = torch.linspace(20.0, 160.0, self.num_mfs[1])
        v_centers = torch.linspace(1.5, 8.5, self.num_mfs[2])

        q_sigmas = torch.full((self.num_mfs[0],), 10.0)
        w_sigmas = torch.full((self.num_mfs[1],), 35.0)
        v_sigmas = torch.full((self.num_mfs[2],), 2.0)

        # Learnable premise parameters
        self.centers = nn.ParameterList([
            nn.Parameter(q_centers.clone()),
            nn.Parameter(w_centers.clone()),
            nn.Parameter(v_centers.clone())
        ])
        self.sigmas = nn.ParameterList([
            nn.Parameter(q_sigmas.clone()),
            nn.Parameter(w_sigmas.clone()),
            nn.Parameter(v_sigmas.clone())
        ])

        # ----------------------------------------------------------------------
        # LAYER 4: Consequent Parameters (First-Order Sugeno: p1*x1 + p2*x2 + p3*x3 + r)
        # Dimensions: [num_rules, 4] for [p_q, p_w, p_arr, r_intercept]
        # ----------------------------------------------------------------------
        consequent_init = torch.zeros(self.num_rules, 4)
        # Seed sensible initial values: baseline duration ~25s + modest slopes
        consequent_init[:, 0] = 0.5   # Queue weight
        consequent_init[:, 1] = 0.15  # Wait time weight
        consequent_init[:, 2] = 1.2   # Arrival weight
        consequent_init[:, 3] = 15.0  # Constant bias
        self.consequents = nn.Parameter(consequent_init)

        # Precompute rule grid index combinations (Cartesian product of MFs)
        grid = torch.meshgrid(
            torch.arange(self.num_mfs[0]),
            torch.arange(self.num_mfs[1]),
            torch.arange(self.num_mfs[2]),
            indexing="ij"
        )
        self.register_buffer("rule_indices", torch.stack([g.flatten() for g in grid], dim=1))  # [27, 3]

    def layer1_fuzzification(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Layer 1: Compute Gaussian membership values for each input variable."""
        # x: [batch_size, 3]
        mf_degrees = []
        for i in range(3):
            val = x[:, i:i+1]                                   # [B, 1]
            c = self.centers[i].unsqueeze(0)                    # [1, M_i]
            s = torch.clamp(self.sigmas[i].unsqueeze(0), min=0.5) # [1, M_i]
            mu = torch.exp(-0.5 * ((val - c) / s) ** 2)         # [B, M_i]
            mf_degrees.append(mu)
        return mf_degrees

    def layer2_firing_strength(self, mf_degrees: List[torch.Tensor]) -> torch.Tensor:
        """Layer 2: Compute T-Norm rule firing strengths (Product operator)."""
        # mf_degrees: list of [B, M_i]
        # rule_indices: [num_rules, 3]
        batch_size = mf_degrees[0].shape[0]
        rule_w = torch.ones(batch_size, self.num_rules, device=mf_degrees[0].device)
        for i in range(3):
            # Gather membership for rule's corresponding MF index
            idx = self.rule_indices[:, i]  # [num_rules]
            selected_mu = mf_degrees[i][:, idx]  # [B, num_rules]
            rule_w = rule_w * selected_mu
        return rule_w  # [B, num_rules]

    def layer3_normalization(self, rule_w: torch.Tensor) -> torch.Tensor:
        """Layer 3: Normalize rule firing strengths w_bar = w_i / sum(w_k)."""
        w_sum = torch.sum(rule_w, dim=1, keepdim=True) + 1e-8
        w_bar = rule_w / w_sum
        return w_bar  # [B, num_rules]

    def layer4_consequents(self, x: torch.Tensor, w_bar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Layer 4: Compute individual rule outputs f_i = P_i * [x, 1]^T
        Returns:
            f: [B, num_rules] raw rule outputs
            weighted_f: [B, num_rules] = w_bar * f
        """
        # x_augmented: [B, 4] with [q, w, arr, 1.0]
        x_aug = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)  # [B, 4]
        # Consequents: [num_rules, 4]
        f = torch.matmul(x_aug, self.consequents.t())  # [B, num_rules]
        weighted_f = w_bar * f
        return f, weighted_f

    def layer5_defuzzification(self, weighted_f: torch.Tensor) -> torch.Tensor:
        """Layer 5: Overall output summation y = sum(w_bar_i * f_i)."""
        y = torch.sum(weighted_f, dim=1, keepdim=True)  # [B, 1]
        # Enforce safety physical clamp [10.0, 60.0]
        return torch.clamp(y, 10.0, 60.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward propagation across all 5 ANFIS layers."""
        # Layer 1
        mfs = self.layer1_fuzzification(x)
        # Layer 2
        w = self.layer2_firing_strength(mfs)
        # Layer 3
        w_bar = self.layer3_normalization(w)
        # Layer 4
        _, weighted_f = self.layer4_consequents(x, w_bar)
        # Layer 5
        output = self.layer5_defuzzification(weighted_f)
        return output

    def get_layer_details(self, x: torch.Tensor) -> Dict[str, Any]:
        """Introspection helper exposing intermediate activations of all 5 layers."""
        with torch.no_grad():
            mfs = self.layer1_fuzzification(x)
            w = self.layer2_firing_strength(mfs)
            w_bar = self.layer3_normalization(w)
            f, weighted_f = self.layer4_consequents(x, w_bar)
            out = self.layer5_defuzzification(weighted_f)

            return {
                "layer1_mfs": [m.cpu().numpy().tolist()[0] for m in mfs],
                "layer2_firing_strengths": w.cpu().numpy().tolist()[0],
                "layer3_normalized_weights": w_bar.cpu().numpy().tolist()[0],
                "layer4_rule_outputs": f.cpu().numpy().tolist()[0],
                "layer5_crisp_output": float(out.cpu().numpy()[0, 0])
            }


# ==============================================================================
# 2. HYBRID LEARNING ENGINE (LSE + BACKPROP)
# ==============================================================================

class ANFISTrafficTuner:
    """
    Hybrid Training and Tuning Orchestrator for ANFIS.
    - Forward LSE: Solves optimal linear consequents via Regularized Least Squares
    - Backward GD: Updates nonlinear Gaussian centers and sigmas via Adam/SGD
    """

    def __init__(self, num_mfs: Tuple[int, int, int] = (3, 3, 3), lr_premise: float = 0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ANFISNet(num_mfs_per_input=num_mfs).to(self.device)
        self.lr_premise = lr_premise

    def generate_synthetic_traffic_dataset(self, n_samples: int = 1200, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates comprehensive synthetic traffic dataset representing real-world junction behavior.
        x = [queue in (0, 50), wait in (0, 180), arrival in (0, 10)]
        target = expert ground-truth green duration in [10, 60] with noise
        """
        np.random.seed(seed)
        q = np.random.uniform(0.0, 50.0, size=(n_samples, 1))
        w = np.random.uniform(0.0, 180.0, size=(n_samples, 1))
        arr = np.random.uniform(0.0, 10.0, size=(n_samples, 1))

        # Ground truth non-linear control surface with saturation kinetics
        # Green time needed to discharge queue: t_discharge ~ q / 0.8 veh/s
        # Surge extension for arrivals: arr * 2.2
        # Urgency multiplier for wait starvation
        starvation_factor = 1.0 + (np.maximum(0.0, w - 80.0) / 120.0)
        base_green = 12.0 + (q * 0.75) + (arr * 1.8)
        target = base_green * starvation_factor
        # Add slight realistic detector jitter
        target += np.random.normal(0.0, 1.0, size=(n_samples, 1))
        target = np.clip(target, 10.0, 60.0)

        x_arr = np.hstack([q, w, arr])
        X = torch.tensor(x_arr, dtype=torch.float32)
        Y = torch.tensor(target, dtype=torch.float32)
        return X, Y

    def fit_consequents_lse(self, x: torch.Tensor, y: torch.Tensor, l2_reg: float = 1e-3):
        """
        Forward Pass: Least Squares Estimation (LSE) for Takagi-Sugeno consequents.
        Minimizes ||A * theta - y||_2^2 + lambda * ||theta||_2^2
        """
        self.model.eval()
        with torch.no_grad():
            mfs = self.model.layer1_fuzzification(x)
            w = self.model.layer2_firing_strength(mfs)
            w_bar = self.model.layer3_normalization(w)  # [B, R]

            B, R = w_bar.shape
            x_aug = torch.cat([x, torch.ones(B, 1, device=x.device)], dim=1)  # [B, 4]

            # Construct design matrix A: shape [B, R * 4]
            # For each rule r, block is w_bar[:, r:r+1] * x_aug
            A_blocks = [w_bar[:, r:r+1] * x_aug for r in range(R)]
            A = torch.cat(A_blocks, dim=1)  # [B, R * 4]

            # Regularized Normal Equation: theta = (A^T A + lambda*I)^(-1) A^T y
            AtA = torch.matmul(A.t(), A)
            reg = l2_reg * torch.eye(AtA.shape[0], device=x.device)
            AtY = torch.matmul(A.t(), y)
            theta = torch.linalg.solve(AtA + reg, AtY)  # [R * 4, 1]

            # Reshape into [R, 4] and update model parameter
            new_consequents = theta.view(R, 4)
            self.model.consequents.data.copy_(new_consequents)

    def train_hybrid(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        epochs: int = 35,
        batch_size: int = 64,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Executes Jang's Hybrid Learning Algorithm:
        - Step 1: Fix premise params, solve consequent params via LSE
        - Step 2: Fix consequent params, backpropagate error to tune premise params
        """
        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)

        # Baseline loss
        with torch.no_grad():
            init_pred = self.model(X_train)
            baseline_loss = torch.mean((init_pred - Y_train) ** 2).item()

        # Optimizer for premise parameters only
        premise_params = list(self.model.centers) + list(self.model.sigmas)
        optimizer = optim.Adam(premise_params, lr=self.lr_premise)
        criterion = nn.MSELoss()

        history = []
        n_samples = X_train.shape[0]

        if verbose:
            print("=" * 70)
            print("Starting ANFIS Hybrid Training (LSE Forward + Backpropagation Backward)")
            print(f"Training Samples: {n_samples} | Epochs: {epochs} | Device: {self.device}")
            print(f"Initial Un-tuned MSE Loss: {baseline_loss:.4f}")
            print("=" * 70)

        for epoch in range(1, epochs + 1):
            # 1. Forward LSE on whole dataset (or large batch) to update consequents
            self.fit_consequents_lse(X_train, Y_train)

            # 2. Mini-batch Gradient Descent for premise parameters
            self.model.train()
            permutation = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                indices = permutation[i:i + batch_size]
                bx, by = X_train[indices], Y_train[indices]

                optimizer.zero_grad()
                preds = self.model(bx)
                loss = criterion(preds, by)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_epoch_loss = epoch_loss / max(1, n_batches)
            history.append({"epoch": epoch, "mse_loss": round(avg_epoch_loss, 4)})

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
                print(f"Epoch {epoch:02d}/{epochs:02d} | MSE Loss: {avg_epoch_loss:8.4f} | "
                      f"RMSE: {np.sqrt(avg_epoch_loss):6.3f}s | "
                      f"Reduction: {((baseline_loss - avg_epoch_loss)/baseline_loss)*100:5.2f}%")

        # Final evaluation metrics
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
            print("ANFIS Tuning Completed Successfully!")
            print(f"Final Metrics -> MSE: {mse:.4f} | RMSE: {rmse:.3f}s | MAE: {mae:.3f}s | R^2 Score: {r2:.4f}")
            print("=" * 70)

        return {
            "baseline_loss": baseline_loss,
            "final_mse": round(mse, 4),
            "final_rmse": round(rmse, 4),
            "final_mae": round(mae, 4),
            "r2_score": round(r2, 4),
            "training_history": history
        }

    def predict(self, queue_length: float, waiting_time: float, arrival_rate: float) -> Dict[str, Any]:
        """Performs single inference returning crisp green light extension."""
        self.model.eval()
        # Robust sanitization and physical bounds clamping
        try:
            q = float(queue_length)
            if np.isnan(q) or np.isinf(q): q = 0.0
        except (ValueError, TypeError): q = 0.0

        try:
            w = float(waiting_time)
            if np.isnan(w) or np.isinf(w): w = 0.0
        except (ValueError, TypeError): w = 0.0

        try:
            arr = float(arrival_rate)
            if np.isnan(arr) or np.isinf(arr): arr = 0.0
        except (ValueError, TypeError): arr = 0.0

        q_clamped = float(np.clip(q, 0.0, 50.0))
        w_clamped = float(np.clip(w, 0.0, 180.0))
        arr_clamped = float(np.clip(arr, 0.0, 10.0))

        x_tensor = torch.tensor([[q_clamped, w_clamped, arr_clamped]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(x_tensor)
            details = self.model.get_layer_details(x_tensor)

        crisp_val = float(out.cpu().numpy()[0, 0])
        return {
            "allocated_green_seconds": round(crisp_val, 2),
            "input_metrics": {"queue": q_clamped, "wait": w_clamped, "arrival": arr_clamped},
            "layer_breakdown": details
        }

    def save_model(self, file_path: str = "anfis_traffic_model.pt"):
        """Saves PyTorch weights and network metadata."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "num_mfs": self.model.num_mfs,
            "num_rules": self.model.num_rules
        }, file_path)
        return file_path

    def load_model(self, file_path: str = "anfis_traffic_model.pt"):
        """Loads saved weights."""
        checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()


# ==============================================================================
# SCRIPT ENTRYPOINT FOR STANDALONE EXECUTION
# ==============================================================================

if __name__ == "__main__":
    tuner = ANFISTrafficTuner()
    X, Y = tuner.generate_synthetic_traffic_dataset(n_samples=1000)
    metrics = tuner.train_hybrid(X, Y, epochs=25, verbose=True)
    save_path = tuner.save_model()
    print(f"Saved trained ANFIS model weights to {save_path}")

    # Demo inference
    demo_res = tuner.predict(queue_length=28.0, waiting_time=75.0, arrival_rate=6.2)
    print(f"Sample Inference -> Allocated Green: {demo_res['allocated_green_seconds']} seconds")
