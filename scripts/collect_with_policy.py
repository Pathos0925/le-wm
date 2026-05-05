"""Collect Atari trajectories using a trained LeWM checkpoint as policy.

Mirrors `collect_atari.py` but actions come from CategoricalCEM (with
optional ε-greedy random injection for exploration) instead of pure
random sampling. Used for the Dyna-style iteration loop: data quality
compounds as the policy improves.

The CEM settings exposed here are intentionally light by default
(``--horizon 8 --num-samples 32 --n-iter 2 --topk 4``) — full eval-time
CEM at ~100 ms/step is far too slow for collection at any meaningful
data scale. The trade-off: collected data reflects a slightly weaker
policy than eval-time CEM, but it's still much better than uniform
random and gets us to interesting states (rallies near the ball, paddle
under the ball, etc.) that pure random data only contains by accident.

Example:
    python scripts/collect_with_policy.py \\
        --checkpoint $STABLEWM_HOME/lewm_atari_v4/lewm_atari_v4_weights.pt \\
        --env ALE/Pong-v5 --frames 200000 --eps 0.2 \\
        --out $STABLEWM_HOME/atari_pong_v4policy.h5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atari_env import make_env  # noqa: E402
from atari_io import HDF5EpisodeWriter  # noqa: E402
from eval_atari import CategoricalCEM, FrameHistory  # noqa: E402
from stable_worldmodel.data.utils import get_cache_dir  # noqa: E402
from train_atari import TrainConfig, build_model  # noqa: E402


def collect(
    env_name: str,
    frames: int,
    out_path: Path,
    seed: int,
    checkpoint: Path,
    horizon: int,
    num_samples: int,
    n_iter: int,
    topk: int,
    eps: float,
    device: str,
) -> None:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = TrainConfig(**ckpt["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

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
    H = cfg.history_size

    env = make_env(env_name, seed=seed)
    rng = np.random.default_rng(seed)
    ep_idx = 0
    total_frames = 0
    sum_returns = 0.0
    t0 = time.time()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with HDF5EpisodeWriter(out_path, mode="overwrite") as writer:
        while total_frames < frames:
            obs, _info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            history = FrameHistory(H)

            ep_pixels: list[np.ndarray] = []
            ep_actions: list[np.ndarray] = []
            ep_rewards: list[np.ndarray] = []
            ep_dones: list[np.ndarray] = []
            ep_eps: list[np.ndarray] = []
            ep_steps: list[np.ndarray] = []
            ep_lives: list[np.ndarray] = []

            ep_return = 0.0
            step_idx = 0
            terminated = truncated = False

            # Bootstrap H-1 noops to fill the history buffer (matches eval).
            for _ in range(H - 1):
                obs_np = np.asarray(obs, dtype=np.uint8)
                history.append_frame(obs_np)
                history.append_action(0)
                next_obs, r, terminated, truncated, info = env.step(0)
                ep_pixels.append(obs_np)
                ep_actions.append(np.array([0], dtype=np.int64))
                ep_rewards.append(np.array([r], dtype=np.float32))
                ep_dones.append(np.array([terminated or truncated], dtype=bool))
                ep_eps.append(np.array([ep_idx], dtype=np.int32))
                ep_steps.append(np.array([step_idx], dtype=np.int32))
                ep_lives.append(np.array([int(info.get("lives", 0))], dtype=np.int32))
                obs = next_obs
                ep_return += float(r)
                step_idx += 1
                total_frames += 1
                if terminated or truncated or total_frames >= frames:
                    break

            while not (terminated or truncated) and total_frames < frames:
                obs_np = np.asarray(obs, dtype=np.uint8)
                history.append_frame(obs_np)
                if rng.random() < eps:
                    action = int(env.action_space.sample())
                else:
                    pixels_t, actions_t = history.tensors(device)
                    action = planner.plan(pixels_t, actions_t)
                history.append_action(action)
                next_obs, r, terminated, truncated, info = env.step(action)

                ep_pixels.append(obs_np)
                ep_actions.append(np.array([action], dtype=np.int64))
                ep_rewards.append(np.array([r], dtype=np.float32))
                ep_dones.append(np.array([terminated or truncated], dtype=bool))
                ep_eps.append(np.array([ep_idx], dtype=np.int32))
                ep_steps.append(np.array([step_idx], dtype=np.int32))
                ep_lives.append(np.array([int(info.get("lives", 0))], dtype=np.int32))

                obs = next_obs
                ep_return += float(r)
                step_idx += 1
                total_frames += 1

            if not ep_pixels:
                continue
            ep_data = {
                "pixels": ep_pixels,
                "action": ep_actions,
                "reward": ep_rewards,
                "done": ep_dones,
                "episode_idx": ep_eps,
                "step_idx": ep_steps,
                "lives": ep_lives,
            }
            writer.write_episode(ep_data)
            sum_returns += ep_return
            ep_idx += 1

            elapsed = time.time() - t0
            print(
                f"[ep {ep_idx:5d}] len={step_idx:5d} return={ep_return:+.1f} "
                f"avg_return={sum_returns / ep_idx:+.2f} "
                f"total={total_frames}/{frames} ({total_frames / max(elapsed, 1e-6):.0f} fps)"
            )

    env.close()
    print(
        f"wrote {ep_idx} episodes / {total_frames} frames -> {out_path}  "
        f"avg episode return: {sum_returns / max(ep_idx, 1):+.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="ALE/Pong-v5")
    ap.add_argument("--frames", type=int, default=200_000)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--n-iter", type=int, default=2)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument(
        "--eps", type=float, default=0.2,
        help="Probability of taking a uniform-random action instead of CEM.",
    )
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = ap.parse_args()

    if args.out is None:
        slug = args.env.split("/")[-1].lower().replace("-v5", "").replace("-v4", "")
        out_path = Path(get_cache_dir()) / f"atari_{slug}_policy.h5"
    else:
        out_path = Path(args.out)

    print(
        f"Collecting {args.frames} frames from {args.env} -> {out_path}\n"
        f"  policy: CEM(H={args.horizon}, S={args.num_samples}, "
        f"iter={args.n_iter}, topk={args.topk}) + ε={args.eps}"
    )
    collect(
        env_name=args.env,
        frames=args.frames,
        out_path=out_path,
        seed=args.seed,
        checkpoint=Path(args.checkpoint),
        horizon=args.horizon,
        num_samples=args.num_samples,
        n_iter=args.n_iter,
        topk=args.topk,
        eps=args.eps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
