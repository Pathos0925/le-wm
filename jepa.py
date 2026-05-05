"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        reward_head=None,
        done_head=None,
        value_head=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        # Optional auxiliary heads (Atari / reward-conditioned planning). When
        # left as ``None`` the existing goal-conditioned behavior is preserved
        # bit-for-bit, and these attributes are not registered as submodules.
        self.reward_head = reward_head
        self.done_head = done_head
        self.value_head = value_head

    def predict_reward(self, emb):
        """Apply the reward head to embeddings of shape ``(..., D)``.

        Returns ``None`` when no reward head was provided.
        """
        return None if self.reward_head is None else self.reward_head(emb)

    def predict_done(self, emb):
        """Apply the done head; returns logits."""
        return None if self.done_head is None else self.done_head(emb)

    def predict_value(self, emb):
        return None if self.value_head is None else self.value_head(emb)

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        return cost

    def get_return_cost(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
        gamma: float = 0.99,
        history_size: int = 3,
        use_done: bool = True,
    ):
        """Negative discounted return for each (B, S) action candidate.

        Replaces the goal-conditioned ``criterion`` for reward-driven planning
        (Atari, etc.). Requires a trained ``reward_head``; ``done_head`` and
        ``value_head`` are used when present.

        ``info_dict["pixels"]`` shape ``(B, S, H, C, H_img, W_img)`` — the
        history window is encoded once and shared across the S candidates.
        ``action_candidates`` shape ``(B, S, T, K)``: the first ``H-1`` entries
        are the *actually executed* historical actions; position ``H-1`` is the
        first action being optimized (the one we'll execute next); positions
        ``[H, T)`` are the remaining optimized future actions. The number of
        optimized actions is therefore ``T - H + 1``, which equals the number
        of predicted future states.

        Cost = ``-Σ_i γ^i · r̂_i · P(survive_{<i})`` plus an optional value
        bootstrap at the trailing predicted state. Returns a ``(B, S)`` tensor;
        smaller is better, matching the existing solver convention.
        """
        if self.reward_head is None:
            raise RuntimeError(
                "get_return_cost requires a reward_head; train one or fall "
                "back to goal-conditioned get_cost."
            )

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)
        action_candidates = action_candidates.to(device)

        H = info_dict["pixels"].size(2)
        T = action_candidates.size(2)
        plan_len = T - H + 1  # number of predicted future states / optimized actions
        assert plan_len > 0, f"need T >= H, got T={T} H={H}"

        info_dict = self.rollout(info_dict, action_candidates, history_size=history_size)
        # predicted_emb: (B, S, T+1, D) — H encoded + (T-H+1) predicted future states
        preds = info_dict["predicted_emb"]
        plan_emb = preds[:, :, H : T + 1]  # (B, S, plan_len, D)

        plan_r = self.reward_head(plan_emb).squeeze(-1)  # (B, S, plan_len)
        discount = (gamma ** torch.arange(plan_len, device=device)).view(1, 1, -1)

        if use_done and self.done_head is not None:
            # Soft survival: P(not done at step t) accumulated up to but not including t.
            plan_d = torch.sigmoid(self.done_head(plan_emb).squeeze(-1))  # (B, S, plan_len)
            cum = torch.cumprod(1.0 - plan_d, dim=-1)
            survive = torch.cat(
                [torch.ones_like(cum[..., :1]), cum[..., :-1]], dim=-1
            )
            returns = (plan_r * discount * survive).sum(dim=-1)
            tail_survive = cum[..., -1]
        else:
            returns = (plan_r * discount).sum(dim=-1)
            tail_survive = torch.ones_like(returns)

        if self.value_head is not None:
            tail_emb = preds[:, :, T]  # (B, S, D)
            tail_v = self.value_head(tail_emb).squeeze(-1)
            returns = returns + (gamma**plan_len) * tail_survive * tail_v

        return -returns
