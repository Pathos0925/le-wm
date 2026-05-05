
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Checkpoints &amp; Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

> **Fork note:** this repo extends LeWM to play Atari games end-to-end —
> see [Atari Fork](#atari-fork) at the bottom for setup, results, and the
> chain of architectural changes the adaptation required.

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
python train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pretrained LeWM checkpoints for each environment are mirrored on the Hugging Face
Hub (model repos), alongside the datasets (dataset repos) in the same collection:

- [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
- [`quentinll/lewm-cube`](https://huggingface.co/quentinll/lewm-cube)
- [`quentinll/lewm-tworooms`](https://huggingface.co/quentinll/lewm-tworooms)
- [`quentinll/lewm-reacher`](https://huggingface.co/quentinll/lewm-reacher)

The full baseline checkpoint suite (PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM, DINO-WM-noprop)
is available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e):

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

### From the Drive archive

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

Place the extracted files under `$STABLEWM_HOME/` and load via:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

`AutoCostModel` accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

### From the Hugging Face mirror

The HF model repos ship the LeWM checkpoint as a `weights.pt` (state dict) plus a
`config.json` describing the model. Convert once to produce the `_object.ckpt`
that `eval.py` expects:

```bash
# download weights.pt + config.json
hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht

# convert to object checkpoint under $STABLEWM_HOME/pusht/lewm_object.ckpt
python - <<'PY'
import json, torch, stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

src = Path(swm.data.utils.get_cache_dir(), "hf_pusht")
out = Path(swm.data.utils.get_cache_dir(), "pusht", "lewm_object.ckpt")

cfg = json.loads((src / "config.json").read_text())
encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False, use_mask_token=False,
)
mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                    hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**cfg["predictor"]),
    action_encoder=Embedder(**cfg["action_encoder"]),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)
sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
PY
```

After conversion, load via `swm.policy.AutoCostModel('pusht/lewm')` as usual.

## Atari Fork

This fork adapts LeWM from continuous-control goal-reaching to **reward-driven
Atari play**. The core JEPA training loop is unchanged; new code wraps it
with frame-stack input handling, discrete action embedding, reward / done /
value / inverse-dynamics heads, return-based MPC planning, and a
Dreamer-V1-style actor-critic trained on imagined latent rollouts.

### Quick start (CUDA 12.8)

```bash
bash scripts/setup_env.sh    # creates .venv, installs torch+cu128 + atari deps
source .venv/bin/activate
```

The setup script pins `torch==2.11.0+cu128`. The default PyPI wheel is
`+cu130`, which fails on CUDA 12.8 drivers with "NVIDIA driver too old".

### Data → world model → actor → eval

```bash
# 1. Collect 1M random Pong rollouts (~11 min, ~110 MB on disk after zstd).
python scripts/collect_atari.py --env ALE/Pong-v5 --frames 1000000

# 2. Precompute discounted return-to-go for the value head.
python scripts/add_return_to_go.py \
    --path $STABLEWM_HOME/atari_pong_random.h5 --gamma 0.99

# 3. Train the JEPA world model (15 epochs, ~38 min on A100).
python train_atari.py --dataset atari_pong_random --num-actions 6 \
    --epochs 15 --batch-size 256 --output-name lewm_atari_v4

# 4. Train an actor on imagined rollouts of the (frozen) world model.
python train_actor_critic.py \
    --wm-checkpoint $STABLEWM_HOME/lewm_atari_v4/lewm_atari_v4_weights.pt \
    --dataset atari_pong_random --epochs 15 --steps-per-epoch 1500 \
    --output-name lewm_atari_actor_v1

# 5. Eval — pick CEM (slow + accurate) or actor (fast).
python eval_atari.py --policy actor --actor-argmax \
    --actor-checkpoint $STABLEWM_HOME/lewm_atari_actor_v1/lewm_atari_actor_v1_weights.pt \
    --num-episodes 20

python eval_atari.py --policy cem \
    --checkpoint $STABLEWM_HOME/lewm_atari_v4/lewm_atari_v4_weights.pt \
    --num-episodes 20 --horizon 15 --num-samples 256
```

A Dyna-style data refresh (collect with the trained policy, merge, retrain)
is available via `scripts/collect_with_policy.py` + `scripts/merge_datasets.py`.

### Architecture changes

| Component | Original LeWM | Atari fork |
|---|---|---|
| Encoder | ViT-Tiny over 224×224 RGB | DQN-style CNN over 4×84×84 grayscale stacks |
| Image normalization | ImageNet z-score | none (encoder owns `/255`) |
| Projector / pred-proj norm | `BatchNorm1d` | `LayerNorm` (see finding 1) |
| Action embedding | Conv1d over continuous floats | one-hot → MLP (`DiscreteActionEmbedder`) |
| Heads | none (goal-conditioned) | reward, done, value, inverse-dynamics |
| Loss | next-emb MSE + SIGReg | + reward MSE (weighted) + done BCE (weighted) + value MSE + inverse-dynamics CE |
| Planning | CEM over continuous actions toward goal image | `CategoricalCEMSolver` toward predicted return; or Dreamer-style trained actor |

### Results (Pong, 20 episodes, seed=42, max_steps=1500)

| Variant | Mean | Std | SEM | Range | ms / step |
|---|---:|---:|---:|---|---:|
| Random | −20.40 | 0.80 | 0.36 | −21..−19 | 0 |
| v1 (BatchNorm projector) | −21.00 | 0.00 | 0.00 | −21 | 99 |
| v2 (BN→LN + reward weighting) | −21.00 | 0.00 | 0.00 | −21 | 99 |
| v3 (+ inverse-dynamics aux) | −21.00 | 0.00 | 0.00 | −21 | 99 |
| **v4 (+ value head + return-to-go)** | **−15.90** | **3.67** | **0.82** | **−21..−7** | **99** |
| v5 (Dyna iter with v4 collected data) | −16.05 | 2.46 | 0.55 | −20..−10 | 99 |
| Actor v1 (Dreamer, sample) | −15.15 | 2.41 | 0.54 | −19..−10 | 0.7 |
| **Actor v1 (Dreamer, argmax)** | **−14.75** | **2.28** | **0.51** | **−20..−11** | **0.5** |

The actor matches CEM in score and runs ≈200× faster at inference.

### Findings, in order of impact

**1. BatchNorm in the projector masks representational collapse.**
The original LeWM uses `BatchNorm1d` between the encoder and the predictor.
With BN in train mode, per-batch normalization makes the embeddings *look*
Gaussian to SIGReg even when the underlying representation has collapsed.
At eval time running stats expose the truth (in our v1, SIGReg train ≈ 1
but val ≈ 170, and CEM scored exactly −21 every episode). Switching the
projector and pred-proj to `LayerNorm` fixed train/eval consistency and
let SIGReg do its job.

**2. Reward sparsity defeats vanilla MSE / BCE.**
Random Pong has 97.7% zero rewards and 0.1% terminations. Plain MSE/BCE
makes the optimal predictor a constant — the reward head learns nothing
about *which states* score. Per-row weighting (10× for non-zero events)
gives the rare signal a fighting chance.

**3. AdaLN-zero gates collapse to action-blind on Atari.**
The predictor's action input flows through AdaLN-zero blocks whose gates
init to 0. Training opens the gates only if action helps next-emb
prediction. In Pong with frame-skip 4, an action only nudges the paddle
~4 px out of 84 — next-emb is ~99% predictable from history alone, so
the gradient through the gates never accumulates. Result: the predictor
ignores its action input and `get_return_cost` returns near-equal scores
across all 6 actions, so CEM picks argmax-of-uniform = noop every step.
An **inverse-dynamics auxiliary loss** (predict `a_t` from `cat(z_t, z_{t+1})`)
forces the encoder to encode actions, which in turn forces the predictor
to use them. Action-sensitivity check after this fix:

| | std across actions | std across time | ratio |
|---|---:|---:|---:|
| v2 (action-blind) | 0.004–0.057 | 0.64–0.96 | 0.005–0.09 |
| v3 (action-aware) | 0.44–0.61 | 0.57–0.96 | 0.64–0.78 |

**4. The CEM horizon is too short to see Pong rewards.**
With horizon 15 and frame-skip 4, an action sequence rarely produces a
reward inside the planning window. The cost is dominated by short-term
reward predictions that are nearly equal across actions, and CEM
converges to a deterministic loop that drifts the paddle to an extreme
and gets it stuck. Training a **value head on precomputed return-to-go**
(`G_t = Σ γ^k r_{t+k}` over each episode) gives MPC a long-horizon
bootstrap: `cost = -Σ γ^t r̂_t (1−d̂)_{<t} − γ^H V(ẑ_H)`. This is the
single biggest score improvement in the chain (−21 → −15.9).

**5. The actor matches CEM, 200× faster, with one important caveat.**
A Dreamer-style actor trained on imagined latent rollouts of the frozen
world model matches CEM in score and runs in ~0.5 ms / step (vs CEM's
~100 ms). Training is fast (~22 min). **Caveat:** imagined returns at
the end of training inflate to +8.7 while real return-to-go data lies
in [−3.4, +1.6] — the actor partly *exploits* the value head, finding
latent states it overestimates. The policy that emerges still plays
real Pong (no swept −21 games in 20 evals), but on harder games this
is likely the next failure mode to fix. Tighter return regularization
(target normalization, value smoothing, model-based penalties) is left
as future work.

### What's *not* in here

- **Online learning.** Everything is offline + one Dyna iteration. The
  pieces are there for a full online loop (see `collect_with_policy.py`).
- **Generalization.** Verified only on Pong. The recipe should work on
  paddle-style and dense-reward games (Breakout, Space Invaders, ...)
  without big changes; exploration-heavy games (Montezuma's Revenge,
  Pitfall) would need richer initial data.
- **Value-head exploitation guards.** See finding 5.
- **Hyperparameter tuning.** Numbers above are first-attempt; a sweep
  over horizon, embed_dim, lr, etc. could move the headline number
  further. Pong remains a *losing* score (random opponent still wins
  most points); positive return is the next milestone.

### Files added by the fork

```
atari_env.py                        shared make_env factory
atari_io.py                         HDF5 writer + dataset merger
actor.py                            CategoricalActor + imagine() + λ-returns
train_atari.py                      JEPA + heads, plain PyTorch trainer
train_actor_critic.py               Dreamer-style actor + critic on imagined rollouts
eval_atari.py                       random / cem / cem_eps / actor eval loop
scripts/setup_env.sh                cu128 venv setup
scripts/collect_atari.py            random-policy data collector
scripts/collect_with_policy.py      trained-policy data collector (CEM + ε)
scripts/add_return_to_go.py         value-head target precomputation
scripts/merge_datasets.py           multi-file HDF5 concat
results/                            per-milestone eval logs
```

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
