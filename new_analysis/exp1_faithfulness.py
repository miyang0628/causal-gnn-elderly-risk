# Experiment 1 — Faithfulness via edge occlusion.
# Question: are the discovered causal edges FUNCTIONALLY used by the model, or
# are they decorative? We take each fold's trained causal model (fixed weights)
# and, at inference only, keep edges in order of cross-household stability
# ("stable-first") vs a random order, measuring AUC as a function of the
# fraction of edges retained. A faithful causal graph degrades more gracefully
# when stable edges are kept and collapses faster when they are removed.
# Comprehensiveness = AUC(full) - AUC(remove top-k stable edges).
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
from pathlib import Path
from common import (load_data, ProposedModel, standardize, fold_causal, fold_indices,
                    eval_model, household_links, TAB_DIR, PROC_DIR, SEED)

OUT = Path(__file__).resolve().parent.parent / "results" / "tables"
from common import MODELS_DIR as MODELS
torch.set_num_threads(4)
d = load_data(); X, y, N, NF = d["X"], d["y"], d["N"], d["N_FOLDS"]
folds_csv = pd.read_csv(PROC_DIR / "household_folds.csv")
hh_fold = dict(zip(folds_csv.HH_ID, folds_csv.fold))
HL = household_links()

def train_freq_for_edge(f):
    """Cross-household frequency of each directed pair among TRAIN households of fold f."""
    train_hh = [h for h in HL if hh_fold.get(h) != f]
    cnt = np.zeros((N, N))
    for h in train_hh:
        seen = set((i, j) for (i, j, t) in HL[h])
        for (i, j) in seen:
            cnt[i, j] += 1
    return cnt / max(len(train_hh), 1)      # freq[src, dst]

def mask_from_edges(edges, N):
    """edges: iterable of (src,dst); build attention mask[target=dst, source=src]."""
    m = np.zeros((N, N), np.float32)
    for (s, t) in edges:
        m[t, s] = 1.0
    return m

rng = np.random.default_rng(SEED)
fracs = [1.0, 0.75, 0.5, 0.25, 0.0]
rows = []
for f in range(NF):
    ckpt = torch.load(MODELS / f"causal_fold{f}.pt", weights_only=False)
    idx_tr, idx_va, idx_te = fold_indices(d, f)
    Xs = standardize(X[idx_tr], X)
    model = ProposedModel(ckpt["mask"], ckpt["lag"], use_lag=True)
    model.load_state_dict(ckpt["state"]); model.eval()
    base_lag = ckpt["lag"].copy()
    # present directed edges (source->target) in this fold's causal graph
    mask0 = ckpt["mask"]            # [target, source]
    present = [(s, t) for t in range(N) for s in range(N) if mask0[t, s] == 1.0]
    freq = train_freq_for_edge(f)
    # order edges by stability (descending)
    stable_order = sorted(present, key=lambda e: freq[e[0], e[1]], reverse=True)
    n_edges = len(present)
    def eval_keep(edges):
        m = mask_from_edges(edges, N)
        lag = base_lag * m           # keep lag only on retained edges
        r = eval_model(model, Xs, y, idx_te, graph=(m, lag))
        return r["auc"]
    for frac in fracs:
        k = int(round(frac * n_edges))
        # keep most-stable-first
        auc_stable = eval_keep(stable_order[:k])
        # keep least-stable-first
        auc_least = eval_keep(stable_order[n_edges - k:])
        # keep random-first (avg over 5 draws)
        aucs_rand = []
        for s in range(5):
            r2 = np.random.default_rng(1000 + s)
            sel = [present[i] for i in r2.choice(n_edges, size=k, replace=False)] if k > 0 else []
            aucs_rand.append(eval_keep(sel))
        rows.append({"fold": f, "frac_kept": frac, "n_kept": k, "n_edges": n_edges,
                     "auc_keep_stable": round(auc_stable, 4),
                     "auc_keep_least": round(auc_least, 4),
                     "auc_keep_random": round(float(np.mean(aucs_rand)), 4),
                     "auc_keep_random_std": round(float(np.std(aucs_rand)), 4)})
        print(f"fold {f} frac={frac:.2f} k={k:>2}/{n_edges} | stable={auc_stable:.3f} "
              f"random={np.mean(aucs_rand):.3f} least={auc_least:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "exp1_faithfulness.csv", index=False)
# summary across folds
summ = df.groupby("frac_kept")[["auc_keep_stable", "auc_keep_random", "auc_keep_least"]].agg(["mean", "std"]).round(4)
print("\n=== Faithfulness: AUC vs fraction of edges kept (mean over folds) ===")
print(summ.to_string())
# comprehensiveness at 25% removed (keep 75% stable vs remove the 25% most stable)
full = df[df.frac_kept == 1.0].groupby("fold").auc_keep_stable.mean()
print("\nSaved exp1_faithfulness.csv")
