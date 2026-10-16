# Experiment 2 — Interpretability, quantified.
# (a) Edge stability: cross-fold Jaccard of the discovered graph vs random graphs.
# (b) Directional determinism: fraction of sensor pairs resolved to a single
#     direction across households (highlighting CO2 <-> TVOC).
# (c) Attention agreement: does an UNCONSTRAINED attention model (same
#     architecture, fully-connected mask) rediscover the causal edges?
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from common import (load_data, ProposedModel, standardize, fold_causal, random_mask,
                    fold_indices, household_links, TAB_DIR, PROC_DIR, SEED)

OUT = Path(__file__).resolve().parent.parent / "results" / "tables"
from common import MODELS_DIR as MODELS
torch.set_num_threads(4)
d = load_data(); X, y, N, NF, FEAT = d["X"], d["y"], d["N"], d["N_FOLDS"], d["FEAT"]
HL = household_links()
res = {}

# ── (a) Edge stability across folds ──────────────────────────
def mask2d_causal(f):
    adj = np.load(PROC_DIR / f"causal_adj_fold{f}.npy")
    return (adj.sum(axis=2) > 0).astype(int)   # [src, dst]
def jaccard(a, b):
    a = a.astype(bool); b = b.astype(bool)
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0
causal_masks = [mask2d_causal(f) for f in range(NF)]
rand_masks = []
for f in range(NF):
    m, _ = random_mask(f, N, seed=SEED + f)   # different random graph per fold
    rand_masks.append((m.T > 0).astype(int))  # back to [src,dst]
cj = [jaccard(causal_masks[a], causal_masks[b]) for a in range(NF) for b in range(a + 1, NF)]
rj = [jaccard(rand_masks[a], rand_masks[b]) for a in range(NF) for b in range(a + 1, NF)]
res["edge_stability"] = {"causal_jaccard_mean": float(np.mean(cj)), "causal_jaccard_std": float(np.std(cj)),
                         "random_jaccard_mean": float(np.mean(rj)), "random_jaccard_std": float(np.std(rj))}
print(f"(a) cross-fold Jaccard: causal={np.mean(cj):.3f}+/-{np.std(cj):.3f}  "
      f"random={np.mean(rj):.3f}+/-{np.std(rj):.3f}")

# ── (b) Directional determinism across households ────────────
# freq of directed pair (collapsed over lag) across all households
cnt = np.zeros((N, N))
for h in HL:
    seen = set((i, j) for (i, j, t) in HL[h])
    for (i, j) in seen:
        cnt[i, j] += 1
freq = cnt / len(HL)
dir_rows = []
for i in range(N):
    for j in range(i + 1, N):
        fwd, bwd = freq[i, j], freq[j, i]
        tot = fwd + bwd
        if tot == 0:
            continue
        dom = max(fwd, bwd) / tot
        dir_rows.append({"pair": f"{FEAT[i]} <-> {FEAT[j]}",
                         "freq_fwd": round(fwd, 3), "freq_bwd": round(bwd, 3),
                         "dominant": FEAT[i] + "->" + FEAT[j] if fwd >= bwd else FEAT[j] + "->" + FEAT[i],
                         "dominance": round(dom, 3)})
ddf = pd.DataFrame(dir_rows).sort_values("dominance", ascending=False)
ddf.to_csv(OUT / "exp2_directionality.csv", index=False)
frac_resolved = float((ddf.dominance >= 0.7).mean())
res["directionality"] = {"n_pairs": int(len(ddf)), "frac_dominance_ge_0.7": frac_resolved,
                         "mean_dominance": float(ddf.dominance.mean())}
co2 = FEAT.index("CO2"); tvoc = FEAT.index("TVOC")
res["co2_tvoc"] = {"freq_CO2_to_TVOC": round(float(freq[co2, tvoc]), 3),
                   "freq_TVOC_to_CO2": round(float(freq[tvoc, co2]), 3)}
print(f"(b) directional pairs: {len(ddf)}, frac dominance>=0.7 = {frac_resolved:.2f}, "
      f"mean dominance={ddf.dominance.mean():.2f}")
print(f"    CO2->TVOC freq={freq[co2,tvoc]:.2f}  TVOC->CO2 freq={freq[tvoc,co2]:.2f}")

# ── (c) Attention agreement (unconstrained FC model) ─────────
# Average learned attention over test samples; test whether attention magnitude
# on a directed pair predicts causal-edge membership and cross-hh frequency.
agg_auc = []; agg_rho = []
for f in range(NF):
    ckpt = torch.load(MODELS / f"fc_fold{f}.pt", weights_only=False)
    idx_tr, idx_va, idx_te = fold_indices(d, f)
    Xs = standardize(X[idx_tr], X)
    model = ProposedModel(ckpt["mask"], ckpt["lag"], use_lag=True)
    model.load_state_dict(ckpt["state"]); model.eval()
    # accumulate attention over a sample of test windows
    import torch as T
    sel = idx_te[:4000]
    with T.no_grad():
        _, att = model(T.tensor(Xs[sel]), return_attn=True)   # (B, heads, N, N) target<-source
    A = att.mean(0).mean(0).numpy()          # [target, source]
    A_src_dst = A.T                          # [source, target] to match freq
    causal2d = mask2d_causal(f)              # [src, dst]
    # off-diagonal entries
    labels = []; scores = []; freqs = []
    for s in range(N):
        for t in range(N):
            if s == t:
                continue
            labels.append(int(causal2d[s, t] > 0))
            scores.append(float(A_src_dst[s, t]))
            freqs.append(float(freq[s, t]))
    if len(set(labels)) > 1:
        agg_auc.append(roc_auc_score(labels, scores))
    rho, _ = spearmanr(scores, freqs)
    agg_rho.append(rho)
res["attention_agreement"] = {"auroc_attn_predicts_causal_mean": float(np.mean(agg_auc)),
                              "auroc_std": float(np.std(agg_auc)),
                              "spearman_attn_vs_freq_mean": float(np.nanmean(agg_rho))}
print(f"(c) attention recovers causal edges: AUROC={np.mean(agg_auc):.3f}+/-{np.std(agg_auc):.3f}, "
      f"spearman(attn,freq)={np.nanmean(agg_rho):.3f}")

import json
json.dump(res, open(OUT / "exp2_interpret.json", "w"), indent=2)
print("\nSaved exp2_interpret.json and exp2_directionality.csv")
