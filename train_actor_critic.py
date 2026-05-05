"""Train an actor + critic on imagined latent rollouts of a frozen world model.

Pre-req: a checkpoint produced by ``train_atari.py`` containing encoder,
predictor, action encoder, projector, pred_proj, reward_head, done_head,
value_head, inverse_dynamics_head.

Loop per training step:
  1. Sample a batch of real frame stacks from the dataset; encode them.
  2. Imagine ``--imagine-horizon`` steps forward using the actor (gradient
     flows through log-probs only) and the frozen world model.
  3. Predict reward / done / value at every imagined state under no_grad.
  4. Compute λ-returns and advantages.
  5. Update the actor (policy gradient with optional entropy bonus).
  6. Update the critic (MSE against detached returns).

Eval is plugged in via ``eval_atari.py --policy actor``.

Example:
    python train_actor_critic.py \\
        --wm-checkpoint $STABLEWM_HOME/lewm_atari_v5/lewm_atari_v5_weights.pt \\
        --dataset atari_pong_v5data --epochs 10 --imagine-horizon 15
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from stable_worldmodel.data.dataset import HDF5Dataset
from stable_worldmodel.data.utils import get_cache_dir

from actor import CategoricalActor, imagine, lambda_returns
from module import MLP
from train_atari import TrainConfig, build_model, collate


@dataclass
class ACConfig:
    # World-model checkpoint
    wm_checkpoint: str = ""
    dataset: str = "atari_pong_v5data"

    # Imagination
    imagine_horizon: int = 15
    gamma: float = 0.99
    lam: float = 0.95

    # Loss weights
    entropy_weight: float = 0.003
    critic_weight: float = 1.0

    # Optim
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    weight_decay: float = 1e-3
    grad_clip: float = 1.0

    # Schedule / IO
    epochs: int = 10
    steps_per_epoch: int = 1000
    batch_size: int = 256
    num_workers: int = 8
    train_split: float = 0.95
    seed: int = 0
    log_every: int = 50
    output_name: str = "lewm_atari_actor"


def step_actor_critic(
    wm,
    actor: CategoricalActor,
    critic: torch.nn.Module,
    batch: dict,
    cfg: ACConfig,
    wm_cfg: TrainConfig,
):
    """Single actor-critic update on one batch. Returns metrics dict."""
    K = actor.num_actions
    H = wm_cfg.history_size

    # 1. Encode real states to seed the imagination.
    with torch.no_grad():
        info = wm.encode(batch)
        emb = info["emb"]  # (B, num_steps, D)

    real_actions = batch["action"][:, : H - 1].squeeze(-1).long()
    real_actions_oh = F.one_hot(real_actions, num_classes=K).float()

    # 2. Imagine. log_probs carries gradient; world-model paths are no_grad.
    out = imagine(
        wm,
        actor,
        real_emb=emb,
        real_actions_oh=real_actions_oh,
        horizon=cfg.imagine_horizon,
        history_size=H,
    )
    next_states = out["next_states"]   # (B, H_im, D)
    act_states = out["act_states"]     # (B, H_im, D)
    log_probs = out["log_probs"]       # (B, H_im)

    # 3. Predict reward / done at each imagined step (no_grad).
    with torch.no_grad():
        r_pred = wm.reward_head(next_states).squeeze(-1)
        d_pred = torch.sigmoid(wm.done_head(next_states).squeeze(-1))

    # 4. Compute λ-returns. Critic is the trainable head; we read it under
    #    no_grad here so the same target is used by both losses.
    with torch.no_grad():
        # Need values at z_t, ẑ_{t+1}, ..., ẑ_{t+H_im}: shape (B, H_im+1, D).
        all_states = torch.cat([act_states[:, :1], next_states], dim=1)
        v_all = critic(all_states).squeeze(-1)
        survive = 1.0 - d_pred  # (B, H_im) — soft "didn't terminate" mask
        G = lambda_returns(r_pred, v_all, survive, cfg.gamma, cfg.lam)
        adv = G - v_all[:, :-1]
        # Standardize advantages for stability.
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

    # 5. Actor loss: policy gradient + entropy bonus.
    # Recompute logits at the actor states (no_grad on states themselves).
    logits = actor(act_states.detach())
    dist = torch.distributions.Categorical(logits=logits)
    log_probs_replay = dist.log_prob(out["actions"])
    entropy = dist.entropy().mean()

    pg_loss = -(adv * log_probs_replay).mean()
    actor_loss = pg_loss - cfg.entropy_weight * entropy

    # 6. Critic loss against detached G.
    v_pred = critic(act_states.detach()).squeeze(-1)
    critic_loss = F.mse_loss(v_pred, G.detach())

    metrics = {
        "actor_loss": actor_loss.item(),
        "pg_loss": pg_loss.item(),
        "entropy": entropy.item(),
        "critic_loss": critic_loss.item(),
        "ret_mean": G.mean().item(),
        "ret_std": G.std().item(),
        "v_mean": v_pred.mean().item(),
        "imagined_r_sum": r_pred.sum(dim=1).mean().item(),
        "p_termination": d_pred.mean().item(),
    }
    return actor_loss, critic_loss, metrics


def main():
    ap = argparse.ArgumentParser()
    cfg_default = ACConfig()
    for k, v in asdict(cfg_default).items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, action="store_true" if not v else "store_false")
        else:
            ap.add_argument(flag, default=v, type=type(v))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    cfg = ACConfig(**{k: getattr(args, k) for k in asdict(cfg_default).keys()})
    if not cfg.wm_checkpoint:
        raise SystemExit("--wm-checkpoint is required")

    torch.manual_seed(cfg.seed)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(get_cache_dir())
    print(f"[init] cache_dir={cache_dir} dataset={cfg.dataset}")
    print(f"[init] wm_checkpoint={cfg.wm_checkpoint}")

    # Load frozen world model.
    wm_ckpt = torch.load(cfg.wm_checkpoint, map_location=args.device, weights_only=False)
    wm_cfg = TrainConfig(**wm_ckpt["config"])
    wm = build_model(wm_cfg).to(args.device)
    wm.load_state_dict(wm_ckpt["model"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)

    # Build trainable actor + critic. Initialize critic from the frozen
    # value_head as a warm start.
    actor = CategoricalActor(
        embed_dim=wm_cfg.embed_dim, num_actions=wm_cfg.num_actions, hidden_dim=wm_cfg.head_hidden
    ).to(args.device)
    critic = MLP(
        input_dim=wm_cfg.embed_dim,
        output_dim=1,
        hidden_dim=wm_cfg.head_hidden,
        norm_fn=torch.nn.LayerNorm,
    ).to(args.device)
    critic.load_state_dict(wm.value_head.state_dict())

    n_actor = sum(p.numel() for p in actor.parameters())
    n_critic = sum(p.numel() for p in critic.parameters())
    print(f"[init] actor={n_actor/1e6:.2f}M  critic={n_critic/1e6:.2f}M params")

    # Dataset (only used as a source of initial states for imagination).
    num_steps = wm_cfg.history_size + wm_cfg.num_preds
    dataset = HDF5Dataset(
        name=cfg.dataset,
        frameskip=1,
        num_steps=num_steps,
        keys_to_load=["pixels", "action", "reward", "done"],
        keys_to_cache=["action", "reward", "done"],
        cache_dir=cache_dir,
    )
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    n_train = int(cfg.train_split * len(dataset))
    n_val = len(dataset) - n_train
    train_set, _val_set = random_split(dataset, [n_train, n_val], generator=rnd_gen)
    train_loader = DataLoader(
        train_set, shuffle=True, drop_last=True, generator=rnd_gen,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        pin_memory=(args.device == "cuda"), collate_fn=collate,
    )
    print(f"[init] dataset clips={len(dataset)} train={n_train} steps_per_epoch={cfg.steps_per_epoch}")

    actor_opt = torch.optim.AdamW(actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.weight_decay)

    out_dir = cache_dir / cfg.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        cfg_out = {**asdict(cfg), "wm_config": asdict(wm_cfg)}
        json.dump(cfg_out, f, indent=2)

    global_step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        loader_iter = iter(train_loader)
        for step_in_epoch in range(cfg.steps_per_epoch):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)
            batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}

            actor_loss, critic_loss, metrics = step_actor_critic(
                wm, actor, critic, batch, cfg, wm_cfg
            )

            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
            actor_opt.step()

            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.grad_clip)
            critic_opt.step()

            global_step += 1
            if global_step % cfg.log_every == 0 or global_step == 1:
                elapsed = time.time() - t0
                msg = (
                    f"actor={metrics['actor_loss']:+.4f} "
                    f"pg={metrics['pg_loss']:+.4f} "
                    f"H(π)={metrics['entropy']:.3f} "
                    f"critic={metrics['critic_loss']:.4f} "
                    f"ret={metrics['ret_mean']:+.3f}±{metrics['ret_std']:.3f} "
                    f"V={metrics['v_mean']:+.3f} "
                    f"r_sum_im={metrics['imagined_r_sum']:+.3f} "
                    f"p_term={metrics['p_termination']:.3f}"
                )
                print(
                    f"[ep {epoch} step {global_step}] {msg}  "
                    f"({global_step / max(elapsed, 1e-6):.1f} step/s)"
                )

        torch.save(
            {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "config": asdict(cfg),
                "wm_config": asdict(wm_cfg),
                "wm_checkpoint": cfg.wm_checkpoint,
            },
            out_dir / f"{cfg.output_name}_weights.pt",
        )

    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
