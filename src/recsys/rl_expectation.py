"""Exact one-step expected-RM objective; no action sampling or PPO clipping."""

import torch
from torch.nn import functional as F

from .data import FIRST_ITEM
from .models import exact_kl


def full_catalog_loss(policy, reference, reward_model, tokens, targets, settings):
    if settings["horizon"] != 1 or settings["group_size"] != 1:
        raise ValueError("Full catalog expectation requires H=1 and group=1")
    if settings.get("reinforce_weight", 1.) != 1. or settings["baseline_mode"] != "ema":
        raise ValueError("Full catalog objective uses its exact reward gradient, without EMA/LOO action baseline")
    if settings["repeat_penalty"] != 0:
        raise ValueError("This exact-expectation study uses no repeat penalty")
    logits = policy(tokens)
    probabilities = logits[:, FIRST_ITEM:].float().softmax(dim=-1)
    with torch.no_grad():
        rewards = reward_model.score_catalog(tokens, settings.get("catalog_chunk_size", 512)).float()
        reference_logits = reference(tokens)
    if rewards.shape != probabilities.shape:
        raise ValueError("RM catalog and policy vocabulary differ")
    expected = (probabilities * rewards).sum(dim=-1).mean()
    kl = exact_kl(logits, reference_logits).mean()
    ce = F.cross_entropy(logits, targets)
    loss = -expected + settings["beta"] * kl + settings["ce_weight"] * ce
    return loss, {"objective_return": float((expected - settings["beta"] * kl).detach()),
        "raw_reward": float(expected.detach()), "penalized_reward": float(expected.detach()),
        "kl": float(kl.detach()), "anchor_ce": float(ce.detach()),
        "full_catalog_items": rewards.shape[1], "reinforce_weight": 1.,
        "action_sampling": False}
