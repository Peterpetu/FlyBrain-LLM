"""Expected-free-energy decoding (AX5): emitting a token is an action.

For each candidate next token c (top-k under the model's prediction), the expected free
energy combines
    pragmatic value : -log p(c | prior)                    (prefer what the model predicts)
    epistemic value : information gain about the latent state if c were observed,
                      approximated by the drop in free energy after inferring on c:
                      IG(c) = F(s_init; c) - F(s_inferred; c)   (how much c would let the state settle)
and the policy emits argmin_c [ -log p(c) - beta_e * IG(c) ] (or samples from softmax(-G / T)).
With beta_e = 0 this is greedy decoding. Costs k extra inference passes per emitted token.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def _prior_and_state(model, st):
    mu, mu1, mu3, s3, pi_prior = model._priors(st)
    logits = model.out(model.out_norm(mu))
    return mu, pi_prior, logits


def info_gain(model, st, mu, pi_prior, cand: torch.Tensor) -> torch.Tensor:
    """Free-energy drop from inference if `cand` (B,) were observed. Returns (B,)."""
    cfg = model.cfg
    with torch.no_grad():
        gamma = model._arousal(st)
    pi0 = torch.exp(model.log_temp) * (gamma if cfg.arousal in ("both", "pi0", "learned") else 1.0)   # same pi_0 as FlyAI.step
    with torch.no_grad():
        c_nll = 0.5 * torch.linalg.matrix_norm(model.out.weight, 2) ** 2
    step = (torch.exp(model.log_eta) / (pi0 * c_nll + pi_prior.mean(-1) + 1.0))[:, None]
    with torch.enable_grad():
        e = model.emb(cand)
        s = (mu + model.amort(torch.cat([e, mu], -1))).detach().requires_grad_(True)
        F0 = model._F(s, mu, cand, pi0, pi_prior)
        for _ in range(max(1, cfg.K_max)):                                    # fixed K steps (no adaptive acceptance): the drop is the proxy
            Fc = model._F(s, mu, cand, pi0, pi_prior)
            g, = torch.autograd.grad(Fc.sum(), s)
            s = (s - step * g).detach().requires_grad_(True)
        F1 = model._F(s, mu, cand, pi0, pi_prior)
    return (F0 - F1).detach()


@torch.no_grad()
def efe_decode(model, prefix: torch.Tensor, n_new: int, beta_e: float = 1.0, top_k: int = 8, temperature: float = 0.0):
    """Generate n_new tokens after `prefix` (B, T0) with EFE decoding. Returns (B, n_new)."""
    was_training = model.training; model.eval(); B = prefix.shape[0]
    st = model.init_state(B, model.emb.weight)
    for t in range(prefix.shape[1]):
        _, st, _ = model.step(st, prefix[:, t])
    out = []
    for _ in range(n_new):
        mu, pi_prior, logits = _prior_and_state(model, st)
        logp = F.log_softmax(logits, -1)
        top_lp, top_idx = logp.topk(top_k, -1)                                      # (B, k)
        G = -top_lp.clone()
        if beta_e != 0.0:
            ig = torch.stack([info_gain(model, st, mu, pi_prior, top_idx[:, j]) for j in range(top_k)], 1)
            G = G - beta_e * ig
        choice = G.argmin(-1) if temperature <= 0 else torch.multinomial(torch.softmax(-G / temperature, -1), 1).squeeze(-1)
        tok = top_idx.gather(1, choice[:, None]).squeeze(1)
        _, st, _ = model.step(st, tok)
        out.append(tok)
    model.train(was_training)
    return torch.stack(out, 1)
