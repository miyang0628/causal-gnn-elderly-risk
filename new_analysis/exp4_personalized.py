# Experiment 4 — Personalized / adaptive causal graphs.
# The ablation showed the GLOBAL causal graph gives no accuracy edge. Here we ask
# whether matching the graph to the individual household helps. On each fold's
# trained causal model we swap the inference-time graph per test household:
#   global      : the fold's train-only causal graph (= proposed; baseline)
#   oracle      : the household's OWN discovered graph (upper bound; uses test
#                 structure, so leaky — reported only as attainable headroom)
#   adaptive    : aggregated graph of the k most similar TRAINING households,
#                 matched by covariate similarity (sensor mean/std), label-free
#                 and deployable.
# Per-household AUC is compared across the three graph sources.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from common import (load_data, ProposedModel, standardize, fold_causal, fold_indices,
                    household_links, links_to_adj, adj_to_mask_lag, TAU_MAX, PROC_DIR, SEED)

OUT = Path(__file__).resolve().parent.parent / "results" / "tables"
from common import MODELS_DIR as MODELS
torch.set_num_threads(4)
d = load_data(); X, y, hh, N, NF = d["X"], d["y"], d["hh"], d["N"], d["N_FOLDS"]
HL = household_links()
folds_csv = pd.read_csv(PROC_DIR / "household_folds.csv")
hh_fold = dict(zip(folds_csv.HH_ID, folds_csv.fold))

def own_graph(h):
    if h not in HL:
        return None
    adj = links_to_adj([HL[h]], N, 0.50)   # single hh -> links present (freq=1>=.5)
    return adj_to_mask_lag(adj, N)

def covariate_vector(h, Xstd):
    """label-free signature of a household: per-sensor mean & std of standardized windows."""
    m = (hh == h)
    sub = Xstd[m]                           # (n,N,W)
    return np.concatenate([sub.mean(axis=(0, 2)), sub.std(axis=(0, 2))])

K_ADAPT = 5
rows = []
for f in range(NF):
    ckpt = torch.load(MODELS / f"causal_fold{f}.pt", weights_only=False)
    idx_tr, idx_va, idx_te = fold_indices(d, f)
    Xs = standardize(X[idx_tr], X)
    model = ProposedModel(ckpt["mask"], ckpt["lag"], use_lag=True)
    model.load_state_dict(ckpt["state"]); model.eval()
    g_global = (ckpt["mask"], ckpt["lag"])
    train_hh = [h for h in np.unique(hh[idx_tr]) if h in HL]
    train_vecs = {h: covariate_vector(h, Xs) for h in train_hh}

    test_hh = np.unique(hh[idx_te])
    for h in test_hh:
        sel = idx_te[hh[idx_te] == h]
        yy = y[sel]
        if len(np.unique(yy)) < 2:
            continue
        xx = torch.tensor(Xs[sel])
        def auc_with(graph):
            model.set_graph(*graph)
            with torch.no_grad():
                import torch.nn.functional as F
                pr = F.softmax(model(xx), 1)[:, 1].numpy()
            return roc_auc_score(yy, pr)
        a_global = auc_with(g_global)
        og = own_graph(h)
        a_oracle = auc_with(og) if og is not None else np.nan
        # adaptive: nearest K training households by covariate distance
        hv = covariate_vector(h, Xs)
        dists = sorted(train_hh, key=lambda th: np.linalg.norm(train_vecs[th] - hv))
        neigh = dists[:K_ADAPT]
        adj_ad = links_to_adj([HL[th] for th in neigh], N, 0.50)
        a_adapt = auc_with(adj_to_mask_lag(adj_ad, N))
        rows.append({"fold": f, "HH_ID": h, "n": len(sel),
                     "auc_global": round(a_global, 4),
                     "auc_oracle": round(a_oracle, 4) if a_oracle == a_oracle else np.nan,
                     "auc_adaptive": round(a_adapt, 4)})
        print(f"fold {f} {h}: global={a_global:.3f} oracle={a_oracle:.3f} adaptive={a_adapt:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "exp4_personalized.csv", index=False)
from scipy import stats
def summ(col):
    v = df[col].dropna()
    return v.mean(), v.std(), len(v)
print("\n=== Per-household AUC by graph source (inference-time swap) ===")
for c in ["auc_global", "auc_oracle", "auc_adaptive"]:
    m, s, n = summ(c)
    print(f"  {c:<14} {m:.3f} +/- {s:.3f}  (n={n})")
# paired tests vs global on common households
common = df.dropna(subset=["auc_oracle"])
t_o = stats.ttest_rel(common.auc_oracle, common.auc_global)
t_a = stats.ttest_rel(df.auc_adaptive, df.auc_global)
print(f"  oracle   vs global: dAUC={ (common.auc_oracle-common.auc_global).mean():+.3f}  p={t_o.pvalue:.3f}")
print(f"  adaptive vs global: dAUC={ (df.auc_adaptive-df.auc_global).mean():+.3f}  p={t_a.pvalue:.3f}")
print("Saved exp4_personalized.csv")
