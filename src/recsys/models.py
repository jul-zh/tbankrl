"""GPT-2 architecture from scratch; IDs are tokens, no pretrained downloads."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2Model

from .data import PAD, BOS, FIRST_ITEM


class Policy(nn.Module):
    def __init__(self, vocab_size, context, hidden=128, layers=2, heads=4, dropout=.1):
        super().__init__()
        self.backbone = GPT2Model(GPT2Config(
            vocab_size=vocab_size, n_positions=context, n_ctx=context,
            n_embd=hidden, n_layer=layers, n_head=heads,
            resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout,
            bos_token_id=BOS, eos_token_id=None, pad_token_id=PAD, use_cache=False,
        ))

    def forward(self, tokens):
        mask = tokens.ne(PAD)
        if not mask.any(dim=1).all():
            raise ValueError("Every context must contain at least one non-padding token")
        hidden = self.backbone(input_ids=tokens, attention_mask=mask, use_cache=False).last_hidden_state
        # Right padding: compute only the last valid position's tied LM head.
        last = hidden[torch.arange(len(tokens), device=tokens.device), mask.sum(dim=1) - 1]
        logits = F.linear(last, self.backbone.wte.weight).float()
        return logits.masked_fill(torch.arange(logits.shape[-1], device=tokens.device) < FIRST_ITEM, -1e9)


class RewardModel(nn.Module):
    def __init__(self, vocab_size, hidden=64, item_only=False):
        super().__init__()
        self.item_only = item_only
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=PAD)
        self.head = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, tokens, candidate):
        mask = tokens.ne(PAD) & tokens.ne(BOS)
        pooled = (self.embedding(tokens) * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        if self.item_only:
            pooled = torch.zeros_like(pooled)
        item = self.embedding(candidate)
        features = torch.cat([pooled, item, pooled * item, (pooled - item).abs()], dim=-1)
        return self.head(features).squeeze(-1).sigmoid()

    @torch.no_grad()
    def score_catalog(self, tokens, chunk_size=512):
        """Unchanged RM scores for every real item, reusing the history embedding."""
        if chunk_size < 1:
            raise ValueError("Catalog chunk size must be positive")
        mask = tokens.ne(PAD) & tokens.ne(BOS)
        pooled = (self.embedding(tokens) * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        if self.item_only:
            pooled = torch.zeros_like(pooled)
        result = []
        for start in range(FIRST_ITEM, self.embedding.num_embeddings, chunk_size):
            item = self.embedding.weight[start:start + chunk_size][None].expand(len(tokens), -1, -1)
            history = pooled[:, None].expand_as(item)
            features = torch.cat([history, item, history * item, (history - item).abs()], dim=-1)
            result.append(self.head(features).squeeze(-1).sigmoid().float())
        return torch.cat(result, dim=1)


def append_actions(tokens, actions):
    """Append before padding, and slide only contexts that are already full."""
    lengths = tokens.ne(PAD).sum(dim=1)
    full = lengths == tokens.shape[1]
    result = tokens.clone()
    result[full, :-1] = tokens[full, 1:]
    result[torch.arange(len(tokens), device=tokens.device), lengths.clamp_max(tokens.shape[1] - 1)] = actions
    return result


def exact_kl(logits, reference_logits):
    # Exclude special tokens rather than multiplying zero by undefined log ratios.
    logp = logits[:, FIRST_ITEM:].float().log_softmax(dim=-1)
    logq = reference_logits[:, FIRST_ITEM:].float().log_softmax(dim=-1)
    return (logp.exp() * (logp - logq)).sum(dim=-1)


def rl_loss(policy, reference, reward_model, tokens, beta, repeat_penalty, horizon, gamma, baseline):
    """Unbiased score-function estimator for reward minus visited-state KL.

    At H>1, KL at later states also depends on earlier sampled actions.
    Its detached future costs enter return-to-go; its direct derivative is
    included separately. Ignoring either path gives an incorrect gradient.
    """
    log_probs, rewards, kls = [], [], []
    for _ in range(horizon):
        logits = policy(tokens)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_probs.append(distribution.log_prob(action))
        with torch.no_grad():
            value = reward_model(tokens, action)
            repeat = tokens.eq(action[:, None]).any(dim=1).float()
            rewards.append(value - repeat_penalty * repeat)
            reference_logits = reference(tokens)
        kls.append(exact_kl(logits, reference_logits))
        tokens = append_actions(tokens, action)
    # Absolute discount weights match J = sum gamma^t (reward_t - beta KL_t).
    running = torch.zeros_like(rewards[0])
    returns = []
    for step in reversed(range(horizon)):
        running = running + gamma**step * (rewards[step] - beta * kls[step].detach())
        returns.append(running)
    returns.reverse()
    score_function = sum(-(logp * (ret - baseline).detach()).mean()
                         for logp, ret in zip(log_probs, returns))
    direct_kl = sum(beta * gamma**step * kl.mean() for step, kl in enumerate(kls))
    loss = score_function + direct_kl
    stats = {
        "objective_return": float(returns[0].mean().detach()),
        "penalized_reward": float(torch.stack(rewards).mean()),
        "kl": float(torch.stack(kls).mean().detach()),
    }
    return loss, stats
