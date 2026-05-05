"""LeWM training loop for Atari.

Differences from `train.py` (the original continuous-control trainer):

  - ``AtariCNN`` encoder over ``(4, 84, 84)`` uint8 frame stacks. No ViT, no
    ImageNet normalization, no image transform plumbing — the encoder owns
    its own ``/255`` normalization.
  - ``DiscreteActionEmbedder`` over discrete action ids (Pong has 6 actions).
  - Reward + done losses on top of next-embedding MSE + SIGReg.
  - Plain PyTorch loop (no Lightning / stable_pretraining / Hydra) so the
    Atari path is independently testable. The architecture pieces live in
    ``module.py`` / ``jepa.py`` and can be lifted into the original Lightning
    trainer later if desired.

Example:
    python train_atari.py \
        --dataset atari_pong_random --num-actions 6 \
        --epochs 5 --batch-size 32

The dataset must already be on disk under ``$STABLEWM_HOME``; produce one
with ``python scripts/collect_atari.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from stable_worldmodel.data.dataset import HDF5Dataset
from stable_worldmodel.data.utils import get_cache_dir

from jepa import JEPA
from module import (
    ARPredictor,
    AtariCNN,
    DiscreteActionEmbedder,
    MLP,
    SIGReg,
)


@dataclass
class TrainConfig:
    dataset: str = "atari_pong_random"
    num_actions: int = 6
    image_size: int = 84
    in_channels: int = 4

    # Architecture
    encoder_hidden: int = 256
    embed_dim: int = 128
    history_size: int = 3
    num_preds: int = 1
    pred_depth: int = 4
    pred_heads: int = 8
    pred_mlp_dim: int = 1024
    pred_dim_head: int = 32
    pred_dropout: float = 0.1

    # Auxiliary heads
    head_hidden: int = 256

    # Loss weights
    sigreg_weight: float = 0.09
    reward_weight: float = 1.0
    done_weight: float = 0.5
    # Random-Atari rewards / dones are extremely sparse (~2% non-zero, ~0.1%
    # done in random Pong). Upweight them in the per-row loss so the heads
    # don't degenerate to constant predictors.
    reward_pos_weight: float = 10.0
    done_pos_weight: float = 10.0

    # Optim
    lr: float = 1e-4
    weight_decay: float = 1e-3
    grad_clip: float = 1.0

    # Schedule / IO
    epochs: int = 5
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.95
    seed: int = 0
    log_every: int = 25
    val_every_epoch: int = 1
    output_name: str = "lewm_atari"

    # Smoke testing: cap steps per epoch for quick verification.
    max_steps_per_epoch: int | None = None


def build_model(cfg: TrainConfig) -> JEPA:
    encoder = AtariCNN(
        in_channels=cfg.in_channels,
        hidden_size=cfg.encoder_hidden,
        image_size=cfg.image_size,
    )
    predictor = ARPredictor(
        num_frames=cfg.history_size,
        depth=cfg.pred_depth,
        heads=cfg.pred_heads,
        mlp_dim=cfg.pred_mlp_dim,
        dim_head=cfg.pred_dim_head,
        dropout=cfg.pred_dropout,
        emb_dropout=0.0,
        input_dim=cfg.embed_dim,
        hidden_dim=cfg.encoder_hidden,
        output_dim=cfg.encoder_hidden,
    )
    action_encoder = DiscreteActionEmbedder(
        num_actions=cfg.num_actions, emb_dim=cfg.embed_dim
    )
    # LayerNorm in the projector / pred_proj instead of BatchNorm1d: with BN
    # the train-time batch-stat normalization masks any underlying collapse
    # (SIGReg sees a "Gaussian" batch every step) but at eval time running
    # stats expose it. LayerNorm normalizes per-sample so train/eval agree
    # and SIGReg does its actual job.
    projector = MLP(
        input_dim=cfg.encoder_hidden,
        output_dim=cfg.embed_dim,
        hidden_dim=2 * cfg.encoder_hidden,
        norm_fn=torch.nn.LayerNorm,
    )
    pred_proj = MLP(
        input_dim=cfg.encoder_hidden,
        output_dim=cfg.embed_dim,
        hidden_dim=2 * cfg.encoder_hidden,
        norm_fn=torch.nn.LayerNorm,
    )
    reward_head = MLP(
        input_dim=cfg.embed_dim,
        output_dim=1,
        hidden_dim=cfg.head_hidden,
        norm_fn=torch.nn.LayerNorm,
    )
    done_head = MLP(
        input_dim=cfg.embed_dim,
        output_dim=1,
        hidden_dim=cfg.head_hidden,
        norm_fn=torch.nn.LayerNorm,
    )
    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
        reward_head=reward_head,
        done_head=done_head,
    )


def step(
    model: JEPA,
    sigreg: SIGReg,
    batch: dict,
    cfg: TrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Forward pass for one batch. Returns (total_loss, metrics_dict)."""
    info = model.encode(batch)
    emb = info["emb"]  # (B, T, embed_dim)
    act_emb = info["act_emb"]

    ctx_emb = emb[:, : cfg.history_size]
    ctx_act = act_emb[:, : cfg.history_size]
    tgt_emb = emb[:, cfg.num_preds :]

    pred_emb = model.predict(ctx_emb, ctx_act)
    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    sig_loss = sigreg(emb.transpose(0, 1))

    loss = pred_loss + cfg.sigreg_weight * sig_loss
    metrics = {"pred": pred_loss.item(), "sigreg": sig_loss.item()}

    # Reward / done heads are trained on PREDICTED next-step embeddings, so
    # action choice flows through them at MPC time:
    #   r_head(ẑ_{t+1})  ≈ reward[t]   (reward earned by a_t at z_t)
    #   d_head(ẑ_{t+1})  ≈ done[t]
    # ẑ_{t+1} = predict(z_{≤t}, a_{≤t}) depends on a_t, so different action
    # sequences produce different scores during planning.
    n_pred = pred_emb.size(1)  # number of predicted next-step embeddings
    pred_r = model.predict_reward(pred_emb)
    if pred_r is not None:
        tgt_r = batch["reward"][:, :n_pred].float().squeeze(-1)
        # Weighted MSE so the rare non-zero rewards don't get washed out.
        r_w = torch.where(
            tgt_r.abs() > 0,
            torch.full_like(tgt_r, cfg.reward_pos_weight),
            torch.ones_like(tgt_r),
        )
        reward_loss = ((pred_r.squeeze(-1) - tgt_r).pow(2) * r_w).mean()
        loss = loss + cfg.reward_weight * reward_loss
        metrics["reward"] = reward_loss.item()

    pred_d = model.predict_done(pred_emb)
    if pred_d is not None:
        tgt_d = batch["done"][:, :n_pred].float().squeeze(-1)
        d_w = torch.where(
            tgt_d > 0.5,
            torch.full_like(tgt_d, cfg.done_pos_weight),
            torch.ones_like(tgt_d),
        )
        done_loss = F.binary_cross_entropy_with_logits(
            pred_d.squeeze(-1), tgt_d, weight=d_w
        )
        loss = loss + cfg.done_weight * done_loss
        metrics["done"] = done_loss.item()

    metrics["loss"] = loss.item()
    return loss, metrics


def collate(batch: list[dict]) -> dict:
    """Plain stack over leading dim. Avoids torch's auto-collate quirks on
    tensors of mixed dtype (e.g. uint8 pixels alongside int64 actions)."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    cfg_default = TrainConfig()
    for k, v in asdict(cfg_default).items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, action="store_true" if not v else "store_false")
        elif v is None:
            ap.add_argument(flag, default=None, type=int)
        else:
            ap.add_argument(flag, default=v, type=type(v))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    cfg_kwargs = {
        k: getattr(args, k)
        for k in asdict(cfg_default).keys()
    }
    cfg = TrainConfig(**cfg_kwargs)
    torch.manual_seed(cfg.seed)

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(get_cache_dir())
    print(f"[init] cache_dir={cache_dir} dataset={cfg.dataset}")

    num_steps = cfg.history_size + cfg.num_preds
    dataset = HDF5Dataset(
        name=cfg.dataset,
        frameskip=1,
        num_steps=num_steps,
        keys_to_load=["pixels", "action", "reward", "done"],
        keys_to_cache=["action", "reward", "done"],
        cache_dir=cache_dir,
    )
    print(f"[init] dataset clips={len(dataset)} columns={dataset.column_names}")

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    n_train = int(cfg.train_split * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=rnd_gen)

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(args.device == "cuda"),
        collate_fn=collate,
    )
    train_loader = DataLoader(
        train_set, shuffle=True, drop_last=True, generator=rnd_gen, **loader_kwargs
    )
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kwargs)

    model = build_model(cfg).to(args.device)
    sigreg = SIGReg(knots=17, num_proj=512).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[init] model params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    out_dir = cache_dir / cfg.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    global_step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            if cfg.max_steps_per_epoch is not None and i >= cfg.max_steps_per_epoch:
                break
            batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
            loss, metrics = step(model, sigreg, batch, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            global_step += 1

            if global_step % cfg.log_every == 0 or global_step == 1:
                elapsed = time.time() - t0
                msg = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(
                    f"[ep {epoch} step {global_step}] {msg}  "
                    f"({global_step / max(elapsed, 1e-6):.1f} step/s)"
                )

        if (epoch + 1) % cfg.val_every_epoch == 0:
            model.eval()
            val_acc: dict[str, float] = {}
            n_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {
                        k: v.to(args.device, non_blocking=True) for k, v in batch.items()
                    }
                    _, metrics = step(model, sigreg, batch, cfg)
                    for k, v in metrics.items():
                        val_acc[k] = val_acc.get(k, 0.0) + v
                    n_batches += 1
            if n_batches:
                msg = " ".join(f"{k}={v / n_batches:.4f}" for k, v in val_acc.items())
                print(f"[ep {epoch} VAL]  {msg}")

        torch.save(
            {"model": model.state_dict(), "config": asdict(cfg)},
            out_dir / f"{cfg.output_name}_weights.pt",
        )

    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
