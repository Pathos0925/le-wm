"""Atari evaluation: trained LeWM + CategoricalCEM MPC vs random baseline.

Loads a checkpoint produced by ``train_atari.py`` and rolls episodes,
selecting actions by minimizing :meth:`JEPA.get_return_cost`. Compares
against a random-action baseline so a sanity-pass / regression is easy
to read.

Action-sequence indexing (matches ``get_return_cost`` and the predictor's
autoregressive structure):

  ::

    pixels      : [z_{t-H+1},  ..., z_{t-1}, z_t        ]   length H
    actions     : [a_{t-H+1},  ..., a_{t-1}, a_t,  ..., a_{t+horizon-1}]
                                              ^^^
                                  (first optimized action — the one executed)
    optimized actions: position H-1 plus positions [H, T)  → ``horizon`` total
    T = (H-1) + horizon

Example:
    python eval_atari.py \\
        --checkpoint $STABLEWM_HOME/lewm_atari/lewm_atari_weights.pt \\
        --env ALE/Pong-v5 --num-episodes 5 --policy cem
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from atari_env import make_env
from jepa import JEPA
from train_atari import TrainConfig, build_model


class CategoricalCEM:
    """Discrete CEM planner over categorical action sequences.

    Maintains per-step categorical distributions over actions. Each iteration:
    sample N sequences via Gumbel-max, score each via ``model.get_return_cost``,
    refit distributions to the empirical frequencies of the top-K elites.
    """

    def __init__(
        self,
        model: JEPA,
        num_actions: int,
        horizon: int = 15,
        num_samples: int = 256,
        n_iter: int = 4,
        topk: int = 32,
        gamma: float = 0.99,
        device: str = "cuda",
        seed: int = 0,
    ):
        self.model = model
        self.num_actions = num_actions
        self.horizon = horizon
        self.num_samples = num_samples
        self.n_iter = n_iter
        self.topk = topk
        self.gamma = gamma
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(seed)

    @torch.inference_mode()
    def plan(
        self,
        history_pixels: torch.Tensor,  # (1, H, 4, 84, 84) uint8
        history_actions: torch.Tensor,  # (1, H-1) int64 — past executed actions
    ) -> int:
        K = self.num_actions
        H = history_pixels.size(1)
        T = (H - 1) + self.horizon  # action seq length

        # Replicate history pixels across S samples.
        pixels_BS = history_pixels.unsqueeze(1).expand(
            1, self.num_samples, *history_pixels.shape[1:]
        )

        # One-hot the H-1 historical actions, replicate across S.
        if H - 1 > 0:
            hist_oh = (
                F.one_hot(history_actions.long(), num_classes=K)
                .float()
                .unsqueeze(1)
                .expand(1, self.num_samples, H - 1, K)
                .to(self.device)
            )
        else:
            hist_oh = torch.zeros(
                1, self.num_samples, 0, K, dtype=torch.float32, device=self.device
            )

        # Per-step categorical distributions over the `horizon` optimized actions.
        probs = torch.full((1, self.horizon, K), 1.0 / K, device=self.device)

        for _ in range(self.n_iter):
            log_p = (
                probs.clamp_min(1e-10)
                .log()
                .unsqueeze(1)
                .expand(1, self.num_samples, self.horizon, K)
            )
            u = torch.rand(
                log_p.shape, generator=self.gen, device=self.device
            ).clamp_min(1e-10)
            gumbel = -(-u.log()).log()
            indices = (log_p + gumbel).argmax(dim=-1)  # (1, S, horizon)
            indices[:, 0] = probs.argmax(dim=-1)  # force first sample to argmax

            future_oh = F.one_hot(indices, num_classes=K).float()  # (1, S, horizon, K)
            full_seq = torch.cat([hist_oh, future_oh], dim=2)  # (1, S, T, K)

            cost = self.model.get_return_cost(
                {"pixels": pixels_BS},
                full_seq,
                gamma=self.gamma,
                history_size=H,
            )  # (1, S)

            _, topk_inds = torch.topk(cost, k=self.topk, dim=-1, largest=False)
            elite_oh = future_oh[
                torch.arange(1, device=self.device).unsqueeze(-1),
                topk_inds,
            ]  # (1, topk, horizon, K)
            probs = elite_oh.mean(dim=1)  # (1, horizon, K)

        # The first sampled action (position 0 of horizon) corresponds to a_t,
        # the action we'll actually execute now.
        return int(probs[0, 0].argmax().item())


class FrameHistory:
    """Bounded buffer holding the last H frame stacks and the H-1 actions
    executed between consecutive frame stacks."""

    def __init__(self, H: int):
        self.H = H
        self.frames: deque = deque(maxlen=H)
        self.actions: deque = deque(maxlen=max(H - 1, 0))

    def append_frame(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def append_action(self, action: int) -> None:
        if self.actions.maxlen and self.actions.maxlen > 0:
            self.actions.append(int(action))

    def is_ready(self) -> bool:
        return len(self.frames) == self.H

    def tensors(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = np.stack(list(self.frames), axis=0)  # (H, 4, 84, 84)
        pixels_t = torch.from_numpy(pixels).unsqueeze(0).to(device)
        if len(self.actions) > 0:
            acts_np = np.array(list(self.actions), dtype=np.int64)
            actions_t = torch.from_numpy(acts_np).unsqueeze(0).to(device)
        else:
            actions_t = torch.empty((1, 0), dtype=torch.long, device=device)
        return pixels_t, actions_t


def load_model(ckpt_path: Path, device: str) -> tuple[JEPA, TrainConfig]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = TrainConfig(**ckpt["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def evaluate(
    env_name: str,
    model: Optional[JEPA],
    cfg: Optional[TrainConfig],
    policy: str,
    num_episodes: int,
    horizon: int,
    num_samples: int,
    n_iter: int,
    topk: int,
    seed: int,
    device: str,
    max_episode_steps: int = 5000,
    eps: float = 0.0,
) -> list[float]:
    env = make_env(env_name, seed=seed)
    rng = np.random.default_rng(seed)
    returns: list[float] = []

    H = cfg.history_size if cfg else 3
    planner: Optional[CategoricalCEM] = None
    if policy in ("cem", "cem_eps"):
        assert model is not None and cfg is not None
        planner = CategoricalCEM(
            model=model,
            num_actions=cfg.num_actions,
            horizon=horizon,
            num_samples=num_samples,
            n_iter=n_iter,
            topk=topk,
            device=device,
            seed=seed,
        )

    for ep in range(num_episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        obs, _ = env.reset(seed=ep_seed)
        history = FrameHistory(H)

        ep_return = 0.0
        ep_steps = 0
        terminated = truncated = False
        plan_time = 0.0
        plan_calls = 0

        # Bootstrap: take H-1 noops so the buffer holds H-1 frames + H-1 actions.
        # After the loop the next obs we observe will be the H-th frame.
        for _ in range(H - 1):
            history.append_frame(np.asarray(obs, dtype=np.uint8))
            history.append_action(0)
            obs, r, terminated, truncated, _ = env.step(0)
            ep_return += float(r)
            ep_steps += 1
            if terminated or truncated:
                break

        while not (terminated or truncated) and ep_steps < max_episode_steps:
            history.append_frame(np.asarray(obs, dtype=np.uint8))
            if not history.is_ready():
                action = 0  # safety: should not normally hit
            elif policy == "random":
                action = int(env.action_space.sample())
            elif policy == "cem_eps" and rng.random() < eps:
                # ε-greedy: with prob eps, take a uniformly-random action.
                # Helps escape degenerate fixed points where CEM converges
                # to a single action that produces a stuck paddle position.
                action = int(env.action_space.sample())
            else:
                pixels_t, actions_t = history.tensors(device)
                t0 = time.time()
                action = planner.plan(pixels_t, actions_t)
                plan_time += time.time() - t0
                plan_calls += 1

            history.append_action(action)
            next_obs, r, terminated, truncated, _ = env.step(action)
            obs = next_obs
            ep_return += float(r)
            ep_steps += 1

        returns.append(ep_return)
        plan_avg = (plan_time / plan_calls * 1000.0) if plan_calls else 0.0
        print(
            f"[{policy:6s} ep {ep}] return={ep_return:+.1f} steps={ep_steps} "
            f"plan_t/step={plan_avg:.1f}ms"
        )

    env.close()
    return returns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--env", default="ALE/Pong-v5")
    ap.add_argument(
        "--policy",
        choices=["cem", "cem_eps", "random"],
        default="cem",
        help="cem requires --checkpoint; cem_eps mixes random with CEM (--eps).",
    )
    ap.add_argument(
        "--eps", type=float, default=0.3,
        help="ε for cem_eps: fraction of steps that take a uniform-random action.",
    )
    ap.add_argument("--num-episodes", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--n-iter", type=int, default=4)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-episode-steps", type=int, default=5000)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = ap.parse_args()

    model: Optional[JEPA] = None
    cfg: Optional[TrainConfig] = None
    if args.policy in ("cem", "cem_eps"):
        if args.checkpoint is None:
            raise SystemExit(f"--checkpoint required for --policy {args.policy}")
        model, cfg = load_model(Path(args.checkpoint), args.device)
        n_p = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[init] loaded {args.checkpoint}: {n_p:.1f}M params  H={cfg.history_size}")

    returns = evaluate(
        env_name=args.env,
        model=model,
        cfg=cfg,
        policy=args.policy,
        num_episodes=args.num_episodes,
        horizon=args.horizon,
        num_samples=args.num_samples,
        n_iter=args.n_iter,
        topk=args.topk,
        seed=args.seed,
        device=args.device,
        max_episode_steps=args.max_episode_steps,
        eps=args.eps,
    )
    arr = np.array(returns, dtype=np.float32)
    print(
        f"\n[{args.policy:6s}] mean={arr.mean():+.2f}  std={arr.std():.2f}  "
        f"min={arr.min():+.1f}  max={arr.max():+.1f}  n={len(arr)}"
    )


if __name__ == "__main__":
    main()
