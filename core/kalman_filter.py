import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional
import torch.nn.functional as F


class KalmanFilterCV3D(nn.Module):
    """
    Constant-velocity Kalman filter for 3D translation (x, y, z).

    State: [px, vx, py, vy, pz, vz]^T  (6D)
    Measurement: [px, py, pz]^T        (3D)
    """

    def __init__(
        self,
        q_pos: float,
        q_vel: float,
        r_meas: float,
        freq: float = 30.0,
    ):
        """
        Args:
            q_pos: process noise variance for position
            q_vel: process noise variance for velocity
            r_meas: measurement noise variance
            freq: sampling frequency (Hz)
        """
        super().__init__()
        self.freq = float(freq)
        dt = 1.0 / self.freq

        # State transition matrix A (6x6)
        A = torch.tensor([
            [1, dt, 0,  0,  0,  0],
            [0,  1, 0,  0,  0,  0],
            [0,  0, 1, dt,  0,  0],
            [0,  0, 0,  1,  0,  0],
            [0,  0, 0,  0,  1, dt],
            [0,  0, 0,  0,  0,  1],
        ], dtype=torch.float32)

        # Measurement matrix H (3x6) – we observe px, py, pz only
        H = torch.tensor([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ], dtype=torch.float32)

        I6 = torch.eye(6, dtype=torch.float32)

        # Process noise covariance Q (6x6)
        Q = torch.diag(torch.tensor(
            [q_pos, q_vel, q_pos, q_vel, q_pos, q_vel],
            dtype=torch.float32
        ))

        # Measurement noise covariance R (3x3)
        R = torch.eye(3, dtype=torch.float32) * r_meas

        # Register as buffers so they track device/dtype with .to(...)
        self.register_buffer("A", A)
        self.register_buffer("H", H)
        self.register_buffer("I6", I6)
        self.register_buffer("Q", Q)
        self.register_buffer("R", R)

        self.reset_state()

    def reset_state(self):
        """
        Clear internal Kalman state. Call when starting a new sequence.
        """
        self.x = None  # state: (6,)
        self.P = None  # covariance: (6, 6)

    @torch.no_grad()
    def step(self, z_t: torch.Tensor, visible: bool | torch.Tensor) -> torch.Tensor:
        """
        Process a single timestep.

        Args:
            z_t:      (3,)  measured translation at time t (predicted translation)
            visible:  bool or 0/1 tensor, whether measurement is valid

        Returns:
            filtered position: tensor of shape (3,)
        """
        # Ensure correct device/dtype
        z_t = z_t.to(self.A.device, dtype=self.A.dtype)

        # Convert visibility to Python bool
        if isinstance(visible, torch.Tensor):
            if visible.numel() != 1:
                raise ValueError("visible tensor must be scalar (numel()==1).")
            visible_bool = bool(visible.item())
        else:
            visible_bool = bool(visible)

        # Initialize state from first measurement
        if self.x is None:
            self.x = torch.zeros(6, device=z_t.device, dtype=z_t.dtype)
            self.x[0::2] = z_t  # set px, py, pz
            self.P = torch.eye(6, device=z_t.device, dtype=z_t.dtype) * 1e-3

        A = self.A
        H = self.H
        Q = self.Q
        R = self.R
        I6 = self.I6

        # --- Prediction step ---
        x_pred = A @ self.x            # (6,)
        P_pred = A @ self.P @ A.T + Q  # (6, 6)

        # --- Optional measurement update ---
        if visible_bool:
            # Innovation
            y = z_t - (H @ x_pred)             # (3,)
            S = H @ P_pred @ H.T + R           # (3, 3)
            K = P_pred @ H.T @ torch.linalg.inv(S)  # (6, 3)

            x_new = x_pred + K @ y             # (6,)
            P_new = (I6 - K @ H) @ P_pred      # (6, 6)
        else:
            # No measurement: use prediction only
            x_new = x_pred
            P_new = P_pred

        # Save state
        self.x = x_new
        self.P = P_new

        # Return filtered position (px, py, pz)
        return x_new[0::2]

    @torch.no_grad()
    def forward(self, z: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        """
        Process an entire sequence (offline).

        Args:
            z:       (T, 3) sequence of measured translations
            visible: (T,)   bool / 0-1 tensor, visibility per frame

        Returns:
            (T, 3) filtered positions
        """
        if z.dim() != 2 or z.size(-1) != 3:
            raise ValueError("z must have shape (T, 3)")
        if visible.dim() != 1 or visible.size(0) != z.size(0):
            raise ValueError("visible must have shape (T,)")

        self.reset_state()

        outputs = []
        for t in range(z.size(0)):
            out_t = self.step(z[t], visible[t])
            outputs.append(out_t)
        return torch.stack(outputs, dim=0)


class KalmanFilterCV3DNP:
    """
    Numpy implementation mirroring KalmanFilterCV3D
    State: [px, vx, py, vy, pz, vz]
    Measurement: [px, py, pz]
    """

    def __init__(self, q_pos: float, q_vel: float, r_meas: float, freq: float = 30.0) -> None:
        self.freq = float(freq)
        dt = 1.0 / self.freq

        self.A = np.array(
            [
                [1, dt, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, dt, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
            ],
            dtype=np.float64,
        )
        self.I6 = np.eye(6, dtype=np.float64)
        self.Q = np.diag(np.array([q_pos, q_vel, q_pos, q_vel, q_pos, q_vel], dtype=np.float64))
        self.R = np.eye(3, dtype=np.float64) * float(r_meas)
        self.reset_state()

    def reset_state(self) -> None:
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None

    def step(self, z_t: np.ndarray, visible: bool) -> np.ndarray:
        z_t = np.asarray(z_t, dtype=np.float64).reshape(3)
        visible_bool = bool(visible)

        if self.x is None:
            self.x = np.zeros(6, dtype=np.float64)
            self.x[0::2] = z_t
            self.P = np.eye(6, dtype=np.float64) * 1e-3

        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q

        if visible_bool:
            y = z_t - (self.H @ x_pred)
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T @ np.linalg.inv(S)
            x_new = x_pred + K @ y
            P_new = (self.I6 - K @ self.H) @ P_pred
        else:
            x_new = x_pred
            P_new = P_pred

        self.x = x_new
        self.P = P_new
        return x_new[0::2].copy()
