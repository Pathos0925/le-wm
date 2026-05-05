"""Actor + imagined-rollout helpers for actor-critic training in latent space.

The world model (encoder, predictor, reward / done / value heads) is
frozen and used to dream forward from initial states sampled from the
real-data buffer. The actor proposes actions inside the dream; gradients
to the actor flow through the discrete-action log-probability term of
the policy gradient. The critic is updated against bootstrapped λ-style
returns computed from the imagined trajectory.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalActor(nn.Module):
    """Stateless policy over the world-model's latent embedding.

    Input:  z of shape ``(..., embed_dim)``.
    Output: action logits of shape ``(..., num_actions)``.
    """

    def __init__(self, embed_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


@torch.no_grad()
def _imagine_one_step(
    model,
    state_seq: torch.Tensor,
    action_seq: torch.Tensor,
    history_size: int,
) -> torch.Tensor:
    """One predictor step under torch.no_grad: returns z_{t+1} of shape (B, D).

    state_seq:  (B, t, D)            grows by 1 per call
    action_seq: (B, t, num_actions)  grows by 1 per call (last entry is the action just taken)
    """
    HS = history_size
    z_trunc = state_seq[:, -HS:]
    a_emb = model.action_encoder(action_seq[:, -HS:])
    z_next = model.predict(z_trunc, a_emb)[:, -1, :]
    return z_next


def imagine(
    model,
    actor: CategoricalActor,
    real_emb: torch.Tensor,        # (B, num_steps, D) — encoded real obs
    real_actions_oh: torch.Tensor, # (B, history_size - 1, K) — past executed actions, one-hot
    horizon: int,
    history_size: int,
):
    """Roll the actor forward `horizon` steps from each row's encoded history.

    The world model is treated as fixed (predict / heads are evaluated under
    no_grad); only ``log_probs`` carry gradient — that's the policy-gradient
    handle. Returns a dict with imagined states and per-step log_probs ready
    for advantage / loss assembly.
    """
    B = real_emb.size(0)
    K = actor.num_actions
    device = real_emb.device

    # Initial state stack & action stack.
    state_seq = real_emb[:, :history_size].clone()        # (B, H, D)
    action_seq = real_actions_oh.clone()                  # (B, H-1, K)

    log_probs: list[torch.Tensor] = []
    actions_taken: list[torch.Tensor] = []
    states_imagined: list[torch.Tensor] = [state_seq[:, -1]]  # z_t

    for _ in range(horizon):
        z = states_imagined[-1].detach()
        logits = actor(z)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()                       # (B,)
        log_p = dist.log_prob(a)                # (B,)
        a_oh = F.one_hot(a, num_classes=K).float()

        # Append action, predict next state under no_grad.
        new_action_seq = torch.cat([action_seq, a_oh.unsqueeze(1)], dim=1)
        z_next = _imagine_one_step(model, state_seq, new_action_seq, history_size)
        state_seq = torch.cat([state_seq, z_next.unsqueeze(1)], dim=1)
        action_seq = new_action_seq

        log_probs.append(log_p)
        actions_taken.append(a)
        states_imagined.append(z_next)

    return {
        # All imagined ẑ_{t+1}, ..., ẑ_{t+horizon}: (B, horizon, D)
        "next_states": torch.stack(states_imagined[1:], dim=1),
        # State the actor *acted from* at each step: z_t, ẑ_{t+1}, ..., ẑ_{t+horizon-1}: (B, horizon, D)
        "act_states": torch.stack(states_imagined[:-1], dim=1),
        "log_probs": torch.stack(log_probs, dim=1),       # (B, horizon)
        "actions": torch.stack(actions_taken, dim=1),     # (B, horizon)
    }


def lambda_returns(
    rewards: torch.Tensor,    # (B, H)
    values: torch.Tensor,     # (B, H+1) — values at z_t, ẑ_{t+1}, ..., ẑ_{t+H}
    survive: torch.Tensor,    # (B, H) — P(not yet terminated) entering each step
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """λ-returns for advantage-style learning.

    G_t = r_t + γ·(1-d_t)·[(1-λ)·V(ẑ_{t+1}) + λ·G_{t+1}],   with G_H = V(ẑ_{t+H}).

    Implemented backwards. ``survive[t]`` here is the per-step factor
    (1 - d_t) for the transition out of step t — i.e. the surviving
    probability that the *next* state was actually reached.
    """
    B, H = rewards.shape
    G = values[:, -1]
    out = torch.zeros_like(rewards)
    for t in range(H - 1, -1, -1):
        bootstrap = (1 - lam) * values[:, t + 1] + lam * G
        G = rewards[:, t] + gamma * survive[:, t] * bootstrap
        out[:, t] = G
    return out
