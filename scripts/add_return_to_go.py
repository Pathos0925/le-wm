"""Compute discounted return-to-go and append it as a new HDF5 column.

For each episode of an existing collected dataset, computes
``G_t = Σ_{k≥0} γ^k r_{t+k}`` where the sum runs to the episode end,
then writes the result as a new resizable column ``return_to_go`` of
shape ``(N_total, 1)`` matching ``reward``.

The column is consumed at training time by the value head: the trained
V(z_t) gives MPC a long-horizon bootstrap so the planner sees beyond
its own rollout horizon.

The script is idempotent — rerun to overwrite an existing column with
a different ``--gamma``.

Example:
    python scripts/add_return_to_go.py \\
        --path $STABLEWM_HOME/atari_pong_random.h5 --gamma 0.99
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np


def compute_returns(rewards: np.ndarray, ep_lens: np.ndarray, ep_offsets: np.ndarray, gamma: float) -> np.ndarray:
    """rewards: (N, 1) or (N,) — discounted-return-to-go per episode."""
    flat = rewards.squeeze(-1) if rewards.ndim == 2 else rewards
    out = np.zeros_like(flat, dtype=np.float32)
    for off, length in zip(ep_offsets, ep_lens):
        end = off + length
        G = 0.0
        # Iterate backward through the episode.
        for t in range(end - 1, off - 1, -1):
            G = float(flat[t]) + gamma * G
            out[t] = G
    return out[:, None].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="HDF5 dataset file to modify in place")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument(
        "--clip-rewards", action="store_true",
        help="Replace the reward column with sign(reward) before computing "
             "return-to-go. Standard DQN-style fix for games with large "
             "reward magnitudes (e.g. MsPacman has rewards up to 800; "
             "without clipping the reward MSE explodes).",
    )
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"no such file: {path}")

    t0 = time.time()
    with h5py.File(path, "r+") as f:
        rewards = f["reward"][:]

        if args.clip_rewards:
            n_pos = int((rewards > 0).sum())
            n_neg = int((rewards < 0).sum())
            print(f"clipping rewards to sign: {n_pos} positive, {n_neg} negative kept (magnitudes lost)")
            clipped = np.sign(rewards).astype(np.float32)
            f["reward"][...] = clipped
            rewards = clipped

        ep_lens = f["ep_len"][:]
        ep_offsets = f["ep_offset"][:]

        n = int(ep_lens.sum())
        print(f"computing return-to-go for {len(ep_lens)} episodes / {n} rows  γ={args.gamma}")

        returns = compute_returns(rewards, ep_lens, ep_offsets, args.gamma)

        if "return_to_go" in f:
            print("overwriting existing 'return_to_go' column")
            del f["return_to_go"]

        compression = hdf5plugin.Blosc(cname="zstd", clevel=3)
        f.create_dataset(
            "return_to_go",
            data=returns,
            maxshape=(None, 1),
            dtype=np.float32,
            chunks=(1, 1),
            **compression,
        )

    elapsed = time.time() - t0
    print(
        f"done in {elapsed:.1f}s  "
        f"min={returns.min():+.3f}  max={returns.max():+.3f}  "
        f"mean={returns.mean():+.3f}  std={returns.std():.3f}"
    )


if __name__ == "__main__":
    main()
