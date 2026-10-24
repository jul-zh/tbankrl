"""On-policy REINFORCE ablations; LOO is a baseline, not PPO/GRPO."""

import torch
from torch.nn import functional as F

from .models import append_actions, exact_kl


def leave_one_out(values, group_size):
    if group_size < 2 or values.numel() % group_size:
        raise ValueError("LOO needs at least two independent trajectories per context")
    grouped = values.detach().reshape(-1, group_size)
    return ((grouped.sum(dim=1, keepdim=True) - grouped) / (group_size - 1)).reshape(-1)


def modified_rl_loss(policy, reference, reward_model, tokens, targets, settings, baseline):
    objective_mode = settings.get("objective_mode", "sampled")
    if objective_mode == "full_catalog":
        from .rl_expectation import full_catalog_loss
        return full_catalog_loss(policy, reference, reward_model, tokens, targets, settings)
    if objective_mode != "sampled":
        raise ValueError("Unknown RL objective mode")
    horizon, group = settings["horizon"], settings["group_size"]
    mode, beta, gamma = settings["baseline_mode"], settings["beta"], settings["gamma"]
    reinforce_weight = settings.get("reinforce_weight", 1.)
    if reinforce_weight not in {0., 1.}:
        raise ValueError("reinforce_weight must be 0 or 1")
    if reinforce_weight == 0 and (horizon != 1 or group != 1 or mode != "ema"):
        raise ValueError("CE/KL control requires H=1, group=1, EMA")
    if mode not in {"ema", "loo"} or horizon < 1 or group < 1:
        raise ValueError("Invalid rollout/baseline settings")
    if (mode == "loo" and group < 2) or (mode == "ema" and group != 1):
        raise ValueError("Use group>=2 for LOO and group=1 for EMA")
    initial = tokens
    tokens = tokens.repeat_interleave(group, dim=0)
    log_probs, rewards, raw_rewards, repeats, kls = [], [], [], [], []
    for _ in range(horizon):
        logits = policy(tokens)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_probs.append(distribution.log_prob(action))
        with torch.no_grad():
            raw = reward_model(tokens, action)
            repeat = tokens.eq(action[:, None]).any(dim=1).float()
            raw_rewards.append(raw)
            repeats.append(repeat)
            rewards.append(raw - settings["repeat_penalty"] * repeat)
            reference_logits = reference(tokens)
        kls.append(exact_kl(logits, reference_logits))
        tokens = append_actions(tokens, action)
    running = torch.zeros_like(rewards[0])
    returns = []
    for step in reversed(range(horizon)):
        # Detached future state costs and direct KL derivatives are both needed.
        running = running + gamma**step * (rewards[step] - beta * kls[step].detach())
        returns.append(running)
    returns.reverse()
    score = sum(-(logp * (ret - (leave_one_out(ret, group) if mode == "loo" else baseline)).detach()).mean()
                for logp, ret in zip(log_probs, returns))
    kl_loss = sum(beta * gamma**step * kl.mean() for step, kl in enumerate(kls))
    loss = score + kl_loss if reinforce_weight else kl_loss
    ce = torch.zeros((), device=initial.device)
    if settings["ce_weight"]:
        ce = F.cross_entropy(policy(initial), targets)
        loss = loss + settings["ce_weight"] * ce
    return loss, {
        "objective_return": float(returns[0].mean().detach()),
        "raw_reward": float(torch.stack(raw_rewards).mean()),
        "penalized_reward": float(torch.stack(rewards).mean()),
        "repeat_fraction": float(torch.stack(repeats).mean()),
        "kl": float(torch.stack(kls).mean().detach()),
        "anchor_ce": float(ce.detach()),
        "reinforce_weight": reinforce_weight,
    }
