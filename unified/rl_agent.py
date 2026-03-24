"""
rl_agent.py
===========
Minimal DQN agent for the 3-action weight-switching problem.

Architecture
------------
  Input  : 6-D congestion observation  (from lidar.congestion_obs)
           + 1 scalar λ (mode parameter, 0=safety, 1=speed)
           → 7-D total input
  Hidden : two layers of 64 units, ReLU
  Output : 3 Q-values, one per weight set

Training details
----------------
  - Replay buffer (deque, capacity BUFFER_SIZE)
  - Target network updated every TARGET_UPDATE steps
  - ε-greedy exploration with linear decay
  - Huber loss (smooth L1)
  - Adam optimiser

The agent is intentionally small.  The observation is only 7-D and
there are only 3 actions, so a large network would overfit.

Usage
-----
    from rl_agent import DQNAgent
    agent = DQNAgent(lam=0.0)          # safety mode
    action = agent.select_action(obs)  # obs shape (6,)
    agent.store(obs, action, reward, next_obs, done)
    agent.train_step()
    agent.save("checkpoints/safety_ep100.pt")
    agent.load("checkpoints/safety_ep100.pt")
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from lidar import OBS_DIM
from weight_sets import N_ACTIONS


# ── Hyper-parameters ──────────────────────────────────────────────────────
BUFFER_SIZE   = 50_000
BATCH_SIZE    = 128
GAMMA         = 0.97        # long episode → keep gamma high
LR            = 3e-4
EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY     = 5_000       # linear decay over this many steps
TARGET_UPDATE = 200         # hard target network copy every N steps
HIDDEN        = 64


class _QNet(nn.Module):
    """Small MLP: (OBS_DIM + 1) → HIDDEN → HIDDEN → N_ACTIONS."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM + 1, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = BUFFER_SIZE) -> None:
        self.buf: deque = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done) -> None:
        self.buf.append((
            np.array(obs,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_obs, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, acts, rews, nobs, dones = zip(*batch)
        return (
            torch.tensor(np.array(obs),  dtype=torch.float32),
            torch.tensor(acts,           dtype=torch.long),
            torch.tensor(rews,           dtype=torch.float32),
            torch.tensor(np.array(nobs), dtype=torch.float32),
            torch.tensor(dones,          dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)


class DQNAgent:
    """
    DQN agent for the 3-action weight-switching problem.

    Parameters
    ----------
    lam : float
        Mode parameter λ ∈ [0, 1].  Appended to every observation so the
        same network can be trained for different modes.
        λ = 0 → safety-first reward shaping
        λ = 1 → speed-first reward shaping
    device : str
        "cpu" or "cuda".  For this problem size, CPU is plenty fast.
    """

    def __init__(self, lam: float = 0.5, device: str = "cpu") -> None:
        self.lam    = float(np.clip(lam, 0.0, 1.0))
        self.device = torch.device(device)

        self.online = _QNet().to(self.device)
        self.target = _QNet().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optim  = optim.Adam(self.online.parameters(), lr=LR)
        self.buffer = ReplayBuffer()

        self.total_steps  = 0
        self.train_steps  = 0
        self._eps         = EPS_START

    # ── Observation helpers ───────────────────────────────────────────────

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        """Append λ to the 6-D congestion obs → 7-D input."""
        return np.append(obs.astype(np.float32), self.lam)

    # ── Action selection ──────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """
        ε-greedy action selection.

        Parameters
        ----------
        obs    : np.ndarray shape (6,) — congestion observation
        greedy : bool — if True, always take the greedy action (evaluation)
        """
        eps = 0.0 if greedy else self._eps
        if random.random() < eps:
            return random.randrange(N_ACTIONS)

        x = torch.tensor(self._augment(obs), dtype=torch.float32,
                          device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.online(x)
        return int(q.argmax(dim=1).item())

    # ── Experience storage ────────────────────────────────────────────────

    def store(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        self.buffer.push(
            self._augment(obs),
            action,
            reward,
            self._augment(next_obs),
            float(done),
        )
        self.total_steps += 1
        # Linear epsilon decay
        progress = min(self.total_steps / EPS_DECAY, 1.0)
        self._eps = EPS_START + progress * (EPS_END - EPS_START)

    # ── Training ──────────────────────────────────────────────────────────

    def train_step(self) -> float | None:
        """
        One gradient step.  Returns the loss value, or None if the buffer
        is too small to sample a full batch.
        """
        if len(self.buffer) < BATCH_SIZE:
            return None

        obs_t, act_t, rew_t, nobs_t, done_t = self.buffer.sample(BATCH_SIZE)
        obs_t  = obs_t.to(self.device)
        act_t  = act_t.to(self.device)
        rew_t  = rew_t.to(self.device)
        nobs_t = nobs_t.to(self.device)
        done_t = done_t.to(self.device)

        # Current Q values
        q_values = self.online(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)

        # Target Q values (using target network)
        with torch.no_grad():
            next_q = self.target(nobs_t).max(dim=1).values
            targets = rew_t + GAMMA * next_q * (1.0 - done_t)

        loss = F.smooth_l1_loss(q_values, targets)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optim.step()

        self.train_steps += 1
        if self.train_steps % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online":       self.online.state_dict(),
            "target":       self.target.state_dict(),
            "optim":        self.optim.state_dict(),
            "total_steps":  self.total_steps,
            "train_steps":  self.train_steps,
            "eps":          self._eps,
            "lam":          self.lam,
        }, path)

    def load(self, path: str | Path) -> None:
        ck = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ck["online"])
        self.target.load_state_dict(ck["target"])
        self.optim.load_state_dict(ck["optim"])
        self.total_steps = ck["total_steps"]
        self.train_steps = ck["train_steps"]
        self._eps        = ck["eps"]
        self.lam         = ck["lam"]

    @property
    def epsilon(self) -> float:
        return self._eps