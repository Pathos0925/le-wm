"""Render a recorded eval episode to an annotated MP4.

Reads an HDF5 trace produced by ``eval_atari.py --record-path`` and writes
an MP4 with the raw game frame on the left, plus a per-step info panel
on the right showing the chosen action, its name, the reward at that
step, the running cumulative return, the model's value estimate, and a
bar chart of the action probability distribution (or one-hot for CEM).

Example:
    python scripts/replay_episode.py \\
        --record /tmp/stablewm/replay_actor.h5 \\
        --episode 0 --env ALE/Pong-v5 \\
        --out /tmp/replay_actor_ep0.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import hdf5plugin  # noqa: F401  registers Blosc/zstd filters used by recorded data
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atari_env import make_env  # noqa: E402


def _action_names(env_name: str, n_actions: int) -> list[str]:
    """Pull action labels from ALE if possible; fall back to indices."""
    try:
        env = make_env(env_name, seed=0)
        names = env.unwrapped.get_action_meanings()
        env.close()
        if len(names) >= n_actions:
            return list(names[:n_actions])
    except Exception:
        pass
    return [f"A{i}" for i in range(n_actions)]


def _draw_panel(
    panel_w: int,
    panel_h: int,
    step: int,
    total_steps: int,
    action: int,
    action_name: str,
    reward: float,
    cum_return: float,
    value: float,
    probs: np.ndarray,
    action_names: list[str],
) -> np.ndarray:
    panel = np.full((panel_h, panel_w, 3), 28, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def line(y, text, color=(230, 230, 230), scale=0.6, thick=1):
        cv2.putText(panel, text, (10, y), font, scale, color, thick, cv2.LINE_AA)

    line(28,  f"step {step:4d} / {total_steps}")
    line(56,  f"action  {action_name}  ({action})", color=(140, 220, 255), scale=0.7, thick=2)
    line(82,  f"reward  {reward:+.0f}",
         color=(140, 255, 140) if reward > 0 else (200, 140, 255) if reward < 0 else (200, 200, 200))
    line(106, f"return  {cum_return:+.1f}",
         color=(140, 255, 140) if cum_return > 0 else (200, 140, 255) if cum_return < 0 else (200, 200, 200))
    line(130, f"V(z)    {value:+.3f}")

    # Action-probability bar chart.
    bar_y0 = 160
    bar_h = 18
    gap = 6
    bar_x0 = 10
    bar_x_max = panel_w - 60
    K = len(probs)
    p = probs.copy()
    if p.sum() <= 0:
        p[action] = 1.0
    p_max = max(p.max(), 1e-6)

    cv2.putText(panel, "policy", (10, bar_y0 - 6), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    for i in range(K):
        y = bar_y0 + i * (bar_h + gap)
        if y + bar_h > panel_h - 4:
            break
        # Label.
        label = f"{action_names[i]:<10}"
        cv2.putText(panel, label, (bar_x0, y + bar_h - 5), font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        # Bar background.
        x_lab = bar_x0 + 92
        cv2.rectangle(panel, (x_lab, y), (bar_x_max, y + bar_h), (60, 60, 60), -1)
        # Filled bar (proportional to prob, scaled to max).
        w = int((bar_x_max - x_lab) * float(p[i]) / p_max)
        color = (140, 220, 255) if i == action else (130, 130, 130)
        cv2.rectangle(panel, (x_lab, y), (x_lab + w, y + bar_h), color, -1)
        # Numeric prob.
        cv2.putText(panel, f"{p[i]:.2f}", (bar_x_max + 4, y + bar_h - 5),
                    font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="HDF5 trace from eval_atari --record-path")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--env", default="ALE/Pong-v5", help="Used only to look up action names.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scale", type=int, default=4, help="Pixel-double factor for the game frame.")
    ap.add_argument("--panel-width", type=int, default=380)
    args = ap.parse_args()

    record = Path(args.record)
    out_path = (
        Path(args.out)
        if args.out
        else record.with_name(f"{record.stem}_ep{args.episode}.mp4")
    )

    with h5py.File(record, "r") as f:
        ep_lens = f["ep_len"][:]
        ep_offsets = f["ep_offset"][:]
        if args.episode < 0 or args.episode >= len(ep_lens):
            raise SystemExit(
                f"--episode {args.episode} out of range (file has {len(ep_lens)} episodes)"
            )
        off = int(ep_offsets[args.episode])
        end = off + int(ep_lens[args.episode])
        frames = f["frames"][off:end]
        actions = f["action"][off:end].squeeze(-1)
        rewards = f["reward"][off:end].squeeze(-1)
        values = f["value"][off:end].squeeze(-1)
        probs_seq = f["action_probs"][off:end]  # (T, K)

    K = probs_seq.shape[1]
    action_names = _action_names(args.env, K)

    fH, fW = frames.shape[1], frames.shape[2]
    game_w = fW * args.scale
    game_h = fH * args.scale
    out_w = game_w + args.panel_width
    out_h = max(game_h, 360)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (out_w, out_h))

    cum_return = 0.0
    T = len(frames)
    for t in range(T):
        cum_return += float(rewards[t])
        # Game pane (RGB → BGR for OpenCV; upscale via nearest neighbor).
        frame_rgb = np.asarray(frames[t], dtype=np.uint8)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_up = cv2.resize(frame_bgr, (game_w, game_h), interpolation=cv2.INTER_NEAREST)
        # Pad if game pane is shorter than out_h.
        if game_h < out_h:
            pad = np.full((out_h - game_h, game_w, 3), 0, dtype=np.uint8)
            frame_up = np.vstack([frame_up, pad])

        # Info panel.
        panel = _draw_panel(
            panel_w=args.panel_width,
            panel_h=out_h,
            step=t,
            total_steps=T,
            action=int(actions[t]),
            action_name=action_names[int(actions[t])],
            reward=float(rewards[t]),
            cum_return=cum_return,
            value=float(values[t]),
            probs=probs_seq[t],
            action_names=action_names,
        )

        full = np.hstack([frame_up, panel])
        writer.write(full)

    writer.release()
    print(f"wrote {out_path}  ({T} frames @ {args.fps} fps  return={cum_return:+.0f})")


if __name__ == "__main__":
    main()
