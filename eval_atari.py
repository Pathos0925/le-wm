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

from actor import CategoricalActor
from atari_env import make_env
from atari_io import HDF5EpisodeWriter
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


def load_actor_bundle(
    actor_ckpt_path: Path, device: str
) -> tuple[JEPA, TrainConfig, CategoricalActor, bool]:
    """Load a trained actor + its frozen world model.

    Returns ``(world_model, wm_cfg, actor, sample_actions)``. The actor
    checkpoint stores the path of the world-model checkpoint it was
    trained against; we resolve it relative to the actor file's parent
    cache_dir if the absolute path no longer exists.
    """
    ckpt = torch.load(actor_ckpt_path, map_location=device, weights_only=False)
    wm_path = Path(ckpt["wm_checkpoint"])
    if not wm_path.exists():
        raise FileNotFoundError(f"linked world-model checkpoint not found: {wm_path}")
    wm_cfg = TrainConfig(**ckpt["wm_config"])
    wm = build_model(wm_cfg).to(device)
    wm_ckpt = torch.load(wm_path, map_location=device, weights_only=False)
    wm.load_state_dict(wm_ckpt["model"])
    wm.eval()

    actor = CategoricalActor(
        embed_dim=wm_cfg.embed_dim,
        num_actions=wm_cfg.num_actions,
        hidden_dim=wm_cfg.head_hidden,
    ).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return wm, wm_cfg, actor, True


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
    actor: Optional[CategoricalActor] = None,
    actor_temperature: float = 1.0,
    actor_argmax: bool = False,
    record_path: Optional[Path] = None,
) -> list[float]:
    env = make_env(
        env_name,
        seed=seed,
        render_mode="rgb_array" if record_path is not None else None,
    )
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    K = cfg.num_actions if cfg is not None else env.action_space.n
    recorder = (
        HDF5EpisodeWriter(record_path, mode="overwrite") if record_path else None
    )
    if recorder is not None:
        recorder.__enter__()

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

        # Per-step recordings (kept only when --record-path is set).
        rec_frames: list[np.ndarray] = []
        rec_actions: list[int] = []
        rec_rewards: list[float] = []
        rec_dones: list[bool] = []
        rec_value: list[float] = []
        rec_probs: list[np.ndarray] = []
        rec_step_idx: list[int] = []

        def record_step(rgb, a, r, d, v, p):
            if recorder is None:
                return
            rec_frames.append(rgb)
            rec_actions.append(int(a))
            rec_rewards.append(float(r))
            rec_dones.append(bool(d))
            rec_value.append(float(v))
            rec_probs.append(p.astype(np.float32))
            rec_step_idx.append(len(rec_step_idx))

        # Bootstrap: take H-1 noops so the buffer holds H-1 frames + H-1 actions.
        for _ in range(H - 1):
            history.append_frame(np.asarray(obs, dtype=np.uint8))
            history.append_action(0)
            if recorder is not None:
                rgb = env.render()
            else:
                rgb = None
            obs, r, terminated, truncated, _ = env.step(0)
            if recorder is not None:
                record_step(rgb, 0, r, terminated or truncated, 0.0, np.zeros(K))
            ep_return += float(r)
            ep_steps += 1
            if terminated or truncated:
                break

        while not (terminated or truncated) and ep_steps < max_episode_steps:
            history.append_frame(np.asarray(obs, dtype=np.uint8))
            v_step = 0.0
            probs_step = np.zeros(K, dtype=np.float32)

            if not history.is_ready():
                action = 0  # safety: should not normally hit
            elif policy == "random":
                action = int(env.action_space.sample())
            elif policy == "cem_eps" and rng.random() < eps:
                action = int(env.action_space.sample())
            elif policy == "actor":
                pixels_t, _ = history.tensors(device)
                t0 = time.time()
                with torch.no_grad():
                    enc_out = model.encode({"pixels": pixels_t})
                    z = enc_out["emb"][:, -1]  # (1, D)
                    logits = actor(z) / actor_temperature
                    if actor_argmax:
                        a = logits.argmax(dim=-1)
                    else:
                        a = torch.distributions.Categorical(logits=logits).sample()
                    action = int(a.item())
                    if recorder is not None:
                        probs_step = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                        if model.value_head is not None:
                            v_step = float(model.value_head(z)[0, 0].item())
                plan_time += time.time() - t0
                plan_calls += 1
            else:
                pixels_t, actions_t = history.tensors(device)
                t0 = time.time()
                action = planner.plan(pixels_t, actions_t)
                plan_time += time.time() - t0
                plan_calls += 1
                if recorder is not None and model is not None:
                    with torch.no_grad():
                        enc_out = model.encode({"pixels": pixels_t})
                        z = enc_out["emb"][:, -1]
                        if model.value_head is not None:
                            v_step = float(model.value_head(z)[0, 0].item())
                    probs_step[action] = 1.0  # CEM is deterministic per call

            history.append_action(action)

            if recorder is not None:
                rgb = env.render()
            else:
                rgb = None

            next_obs, r, terminated, truncated, _ = env.step(action)
            if recorder is not None:
                record_step(rgb, action, r, terminated or truncated, v_step, probs_step)
            obs = next_obs
            ep_return += float(r)
            ep_steps += 1

        if recorder is not None and rec_frames:
            ep_data = {
                "frames": [np.asarray(f, dtype=np.uint8) for f in rec_frames],
                "action": [np.array([a], dtype=np.int64) for a in rec_actions],
                "reward": [np.array([r], dtype=np.float32) for r in rec_rewards],
                "done": [np.array([d], dtype=bool) for d in rec_dones],
                "value": [np.array([v], dtype=np.float32) for v in rec_value],
                "action_probs": [p.astype(np.float32) for p in rec_probs],
                "episode_idx": [np.array([ep], dtype=np.int32) for _ in rec_step_idx],
                "step_idx": [np.array([s], dtype=np.int32) for s in rec_step_idx],
            }
            recorder.write_episode(ep_data)

        returns.append(ep_return)
        plan_avg = (plan_time / plan_calls * 1000.0) if plan_calls else 0.0
        print(
            f"[{policy:6s} ep {ep}] return={ep_return:+.1f} steps={ep_steps} "
            f"plan_t/step={plan_avg:.1f}ms"
        )

    if recorder is not None:
        recorder.__exit__(None, None, None)
    env.close()
    return returns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--env", default="ALE/Pong-v5")
    ap.add_argument(
        "--policy",
        choices=["cem", "cem_eps", "random", "actor"],
        default="cem",
        help="cem/cem_eps require --checkpoint (a world-model ckpt); "
             "actor requires --actor-checkpoint (a train_actor_critic ckpt).",
    )
    ap.add_argument(
        "--eps", type=float, default=0.3,
        help="ε for cem_eps: fraction of steps that take a uniform-random action.",
    )
    ap.add_argument("--actor-checkpoint", type=str, default=None)
    ap.add_argument("--actor-temperature", type=float, default=1.0)
    ap.add_argument(
        "--actor-argmax", action="store_true",
        help="Greedy mode: take argmax of actor logits instead of sampling.",
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
    ap.add_argument(
        "--record-path", type=str, default=None,
        help="If set, dumps an HDF5 trace of every episode (raw RGB frames + "
             "actions + rewards + value + action probs). Pair with "
             "scripts/replay_episode.py to render an annotated MP4.",
    )
    args = ap.parse_args()

    model: Optional[JEPA] = None
    cfg: Optional[TrainConfig] = None
    actor: Optional[CategoricalActor] = None
    if args.policy in ("cem", "cem_eps"):
        if args.checkpoint is None:
            raise SystemExit(f"--checkpoint required for --policy {args.policy}")
        model, cfg = load_model(Path(args.checkpoint), args.device)
        n_p = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[init] loaded {args.checkpoint}: {n_p:.1f}M params  H={cfg.history_size}")
    elif args.policy == "actor":
        if args.actor_checkpoint is None:
            raise SystemExit("--actor-checkpoint required for --policy actor")
        model, cfg, actor, _ = load_actor_bundle(Path(args.actor_checkpoint), args.device)
        n_p = sum(p.numel() for p in model.parameters()) / 1e6
        n_a = sum(p.numel() for p in actor.parameters()) / 1e6
        print(
            f"[init] loaded actor={args.actor_checkpoint}  wm={n_p:.1f}M  actor={n_a:.2f}M  H={cfg.history_size}"
            f"  argmax={args.actor_argmax}  T={args.actor_temperature}"
        )

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
        actor=actor,
        actor_temperature=args.actor_temperature,
        actor_argmax=args.actor_argmax,
        record_path=Path(args.record_path) if args.record_path else None,
    )
    arr = np.array(returns, dtype=np.float32)
    print(
        f"\n[{args.policy:6s}] mean={arr.mean():+.2f}  std={arr.std():.2f}  "
        f"min={arr.min():+.1f}  max={arr.max():+.1f}  n={len(arr)}"
    )


if __name__ == "__main__":
    main()
