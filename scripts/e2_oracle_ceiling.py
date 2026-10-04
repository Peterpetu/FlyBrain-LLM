"""Post-hoc diagnostic (not a hypothesis test): the Bayes-optimal accuracy on E2/E3's held-out streams,
by exact per-key Bayesian filtering with the generator's own rates (p_change, p_outlier, sigma) and the
value codebook. If the learned arms sit at this ceiling, the stream cannot show a state advantage."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyprinc import Vocab, rebind_stream
from flyprinc.tasks import WRITE, QUERY

VOC = Vocab(n_keys=2048, n_vals=64, d_key=64, d_val=16, seed=1234); _, V = VOC.tensors(); V = V.numpy()
REGIMES = [(0.3, 0.0), (1.0, 0.0), (0.3, 0.1), (1.0, 0.1)]; B, N_STREAMS, P, T = 8, 4, 48, 1500


def oracle_stream(stream, sigma, p_out, p_change, rng):
    """Exact HMM filter per key over the 64 candidate current values; returns per-query (correct, tag)."""
    kinds, keys, vals, tags = stream["kinds"], stream["keys"], stream["vals"], stream["tags"]; Bn, Tn = keys.shape; n_vals = V.shape[0]
    noise_sd = sigma / np.sqrt(V.shape[1]); out = []
    for b in range(Bn):
        post = {}; last_t = {}
        for t in range(Tn):
            k = int(keys[b, t])
            if kinds[t] == WRITE:
                obs = V[vals[b, t]] + noise_sd * rng.standard_normal(V.shape[1])              # the probe's observation model
                p = post.get(k, np.full(n_vals, 1.0 / n_vals)); dt = t - last_t.get(k, t)
                stay = (1 - p_change) ** max(dt, 1); p = stay * p + (1 - stay) / n_vals            # change-point prior over the elapsed steps
                ll = -((obs[None, :] - V) ** 2).sum(-1) / (2 * noise_sd ** 2); lik = np.exp(ll - ll.max())
                em = (1 - p_out) * lik + p_out * (lik.sum() - lik) / max(n_vals - 1, 1)             # emission: true value, or an outlier drawn from the others
                p = p * em; p = p / p.sum(); post[k] = p; last_t[k] = t
            else:
                p = post.get(k, np.full(n_vals, 1.0 / n_vals)); out.append((int(np.argmax(p)) == int(vals[b, t]), int(tags[t])))
    return out


def main():
    rng = np.random.default_rng(778); res = {}
    streams = {r: [rebind_stream(B, P, T, VOC, rng, p_change=0.001, p_outlier=r[1]) for _ in range(N_STREAMS)] for r in REGIMES}   # E2/E3's held_out()
    orng = np.random.default_rng(778); cls_all = []; mix_all = []
    for (sig, po), ss in streams.items():
        rows = [r for s in ss for r in oracle_stream(s, sig, po, 0.001, orng)]; ok = np.array([r[0] for r in rows]); tg = np.array([r[1] for r in rows])
        per = {int(t): float(ok[tg == t].mean()) for t in np.unique(tg)}; cls = float(np.mean(list(per.values()))); mix = float(ok.mean())
        res[f"s{sig:g}_o{po:g}"] = {"acc": mix, "per_tag": per, "acc_classes": cls, "n": int(len(ok))}; cls_all.append(cls); mix_all.append(mix)
        print(f"regime sigma={sig} p_out={po}: oracle acc {mix:.4f}, per class {per}, class-balanced {cls:.4f}")
    res["acc_classes"] = float(np.mean(cls_all)); res["acc_mean"] = float(np.mean(mix_all))
    print(f"ORACLE: class-balanced {res['acc_classes']:.4f}, natural mix {res['acc_mean']:.4f}  (E3 learned arms: 0.690 / 0.897)")
    Path("experiments/p3_e3_two_trace/oracle_ceiling.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__": main()
