"""Roll Atari trajectories and dump them to HDF5 in stable_worldmodel's format.

Output schema (per row in the flat HDF5 layout):
  pixels      : (4, 84, 84)  uint8    — 4-frame stack, NCHW
  action      : (1,)         int64    — discrete action id
  reward      : (1,)         float32  — raw env reward (no clipping)
  done        : (1,)         bool     — terminated OR truncated
  episode_idx : (1,)         int32
  step_idx    : (1,)         int32
  lives       : (1,)         int32    — Atari lives counter (for analysis)

The companion `ep_len` / `ep_offset` arrays are written automatically by
HDF5Writer; downstream HDF5Dataset uses them for episode addressing.

Atari preprocessing follows the standard recipe:
  - gym.make(env, frameskip=1)       # disable env-internal frameskip
  - AtariPreprocessing(frame_skip=4, screen_size=84, grayscale_obs=True)
  - FrameStackObservation(stack_size=4)

`-v5` gives sticky-action stochasticity (repeat_action_probability=0.25).

Example:
    python scripts/collect_atari.py --env ALE/Pong-v5 --frames 100000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers compression filters at import)
import numpy as np

# scripts/ is not a package; add the repo root to sys.path so atari_env imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym  # noqa: E402

from atari_env import make_env  # noqa: E402
from stable_worldmodel.data.utils import get_cache_dir  # noqa: E402


class HDF5EpisodeWriter:
    """Minimal append-style writer producing HDF5 files compatible with
    stable_worldmodel.data.dataset.HDF5Dataset.

    Layout: one resizable 1-D-on-axis-0 dataset per column, plus 1-D
    `ep_len` and `ep_offset` metadata. Schema is inferred from the first
    episode and locked thereafter. This is a local stand-in for the
    HDF5Writer in stable_worldmodel git HEAD (not yet on PyPI 0.0.6).
    """

    def __init__(self, path: Path, mode: str = "overwrite") -> None:
        if mode not in ("overwrite", "error", "append"):
            raise ValueError(f"mode must be overwrite|error|append, got {mode}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self._f: h5py.File | None = None
        self._initialized = False
        self._global_ptr = 0

    def __enter__(self) -> "HDF5EpisodeWriter":
        exists = self.path.exists()
        if exists and self.mode == "error":
            raise FileExistsError(self.path)
        if not exists or self.mode == "overwrite":
            self._f = h5py.File(self.path, "w", libver="latest")
        else:
            self._f = h5py.File(self.path, "a", libver="latest")
            if "ep_len" in self._f:
                self._global_ptr = int(self._f["ep_len"][:].sum())
                self._initialized = True
        return self

    def __exit__(self, *exc) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def _init_schema(self, sample_ep: dict) -> None:
        for col, vals in sample_ep.items():
            sample = np.asarray(vals[0])
            self._f.create_dataset(
                col,
                shape=(0, *sample.shape),
                maxshape=(None, *sample.shape),
                dtype=sample.dtype,
                chunks=(1, *sample.shape),
            )
        self._f.create_dataset("ep_len", shape=(0,), maxshape=(None,), dtype=np.int32)
        self._f.create_dataset("ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64)

    def write_episode(self, ep_data: dict) -> None:
        assert self._f is not None, "writer used outside `with` block"
        if not self._initialized:
            self._init_schema(ep_data)
            self._initialized = True

        ep_len = len(next(iter(ep_data.values())))
        for col, vals in ep_data.items():
            ds = self._f[col]
            ds.resize(self._global_ptr + ep_len, axis=0)
            ds[self._global_ptr : self._global_ptr + ep_len] = np.asarray(vals)

        n = self._f["ep_len"].shape[0]
        self._f["ep_len"].resize(n + 1, axis=0)
        self._f["ep_len"][n] = ep_len
        self._f["ep_offset"].resize(n + 1, axis=0)
        self._f["ep_offset"][n] = self._global_ptr

        self._global_ptr += ep_len


def collect(env_name: str, frames: int, out_path: Path, seed: int) -> None:
    env = make_env(env_name, seed=seed)
    rng = np.random.default_rng(seed)
    ep_idx = 0
    total_frames = 0
    t0 = time.time()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with HDF5EpisodeWriter(out_path, mode="overwrite") as writer:
        while total_frames < frames:
            obs, _info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            ep_pixels: list[np.ndarray] = []
            ep_actions: list[np.ndarray] = []
            ep_rewards: list[np.ndarray] = []
            ep_dones: list[np.ndarray] = []
            ep_eps: list[np.ndarray] = []
            ep_steps: list[np.ndarray] = []
            ep_lives: list[np.ndarray] = []

            step_idx = 0
            terminated = truncated = False
            while not (terminated or truncated):
                obs_np = np.asarray(obs, dtype=np.uint8)
                action = int(env.action_space.sample())
                next_obs, reward, terminated, truncated, info = env.step(action)
                lives = int(info.get("lives", 0))

                ep_pixels.append(obs_np)
                ep_actions.append(np.array([action], dtype=np.int64))
                ep_rewards.append(np.array([reward], dtype=np.float32))
                ep_dones.append(np.array([terminated or truncated], dtype=bool))
                ep_eps.append(np.array([ep_idx], dtype=np.int32))
                ep_steps.append(np.array([step_idx], dtype=np.int32))
                ep_lives.append(np.array([lives], dtype=np.int32))

                obs = next_obs
                step_idx += 1
                total_frames += 1
                if total_frames >= frames:
                    break

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
            ep_idx += 1

            elapsed = time.time() - t0
            print(
                f"[ep {ep_idx:5d}] len={step_idx:5d}  "
                f"total={total_frames}/{frames}  "
                f"({total_frames / max(elapsed, 1e-6):.0f} fps)"
            )

    env.close()
    print(f"wrote {ep_idx} episodes / {total_frames} frames -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="ALE/Pong-v5")
    ap.add_argument("--frames", type=int, default=100_000)
    ap.add_argument(
        "--out",
        default=None,
        help="output .h5 path; defaults to "
        "$STABLEWM_HOME/datasets/atari_<game>_random.h5",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.out is None:
        slug = args.env.split("/")[-1].lower().replace("-v5", "").replace("-v4", "")
        out_path = Path(get_cache_dir()) / f"atari_{slug}_random.h5"
    else:
        out_path = Path(args.out)

    print(f"Collecting {args.frames} frames from {args.env} -> {out_path}")
    collect(args.env, args.frames, out_path, args.seed)


if __name__ == "__main__":
    main()
