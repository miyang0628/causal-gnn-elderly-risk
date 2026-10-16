# Experiment 3 — Leakage-injection stress test.
# Demonstrates the leakage-free protocol as an ACTIVE diagnostic: we deliberately
# re-introduce three common leakage modes and measure how much each inflates AUC
# relative to the leakage-free (LF) protocol. All runs use the proposed model.
#   LF   : train-only standardization + household-level split + train-only graph
#   L1   : preprocessing leakage (global standardization over ALL samples)
#   L2   : graph leakage (causal graph built from ALL households incl. test)
#   L3   : record-wise split leakage (windows split ignoring household grouping)
#   ALL  : L1 + L2 + L3 together
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from common import (load_data, ProposedModel, standardize, fold_causal, fold_indices,
                    class_weight, train_model, eval_model, household_links, links_to_adj,
                    adj_to_mask_lag, seed_all, SEED, PROC_DIR)

OUT = Path(__file__).resolve().parent.parent / "results" / "tables"
torch.set_num_threads(4)
d = load_data(); X, y, hh, N, NF = d["X"], d["y"], d["hh"], d["N"], d["N_FOLDS"]
HL = household_links()

# graph built from ALL households (leaky) at freq>=50%
adj_all = links_to_adj(list(HL.values()), N, 0.50)
mask_all, lag_all = adj_to_mask_lag(adj_all, N)

def run_once(Xs, idx_tr, idx_va, idx_te, graph):
    cw = class_weight(y, idx_tr)
    seed_all(SEED); torch.manual_seed(SEED)
    model = ProposedModel(graph[0], graph[1], use_lag=True)
    model = train_model(model, Xs, y, idx_tr, idx_va, cw)
    return eval_model(model, Xs, y, idx_te)["auc"]

rows = []
t0 = time.time()
for f in range(NF):
    idx_tr, idx_va, idx_te = fold_indices(d, f)
    g_lf = fold_causal(f, N)
    Xs_lf = standardize(X[idx_tr], X)                 # train-only
    Xs_global = standardize(X, X)                     # global (leaky)

    # LF
    auc_lf = run_once(Xs_lf, idx_tr, idx_va, idx_te, g_lf)
    # L1 preprocessing leakage
    auc_l1 = run_once(Xs_global, idx_tr, idx_va, idx_te, g_lf)
    # L2 graph leakage
    auc_l2 = run_once(Xs_lf, idx_tr, idx_va, idx_te, (mask_all, lag_all))

    # L3 record-wise split leakage: ignore household grouping entirely.
    # Match test-set size to the household-level fold; stratify by y.
    seed_all(SEED + f)
    n_all = len(y); all_idx = np.arange(n_all)
    tr2, tmp = train_test_split(all_idx, test_size=0.4, random_state=SEED + f, stratify=y)
    va2, te2 = train_test_split(tmp, test_size=0.5, random_state=SEED + f, stratify=y[tmp])
    Xs_rw = standardize(X[tr2], X)                    # still train-only scaling; leakage is the split
    auc_l3 = run_once(Xs_rw, tr2, va2, te2, g_lf)

    # ALL combined
    Xs_rw_global = standardize(X, X)
    auc_all = run_once(Xs_rw_global, tr2, va2, te2, (mask_all, lag_all))

    rows.append({"fold": f, "LF": round(auc_lf, 4), "L1_preproc": round(auc_l1, 4),
                 "L2_graph": round(auc_l2, 4), "L3_recordwise": round(auc_l3, 4),
                 "ALL": round(auc_all, 4)})
    print(f"fold {f} | LF={auc_lf:.3f} L1={auc_l1:.3f} L2={auc_l2:.3f} "
          f"L3={auc_l3:.3f} ALL={auc_all:.3f}  [{time.time()-t0:.0f}s]", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "exp3_leakage.csv", index=False)
m = df[["LF", "L1_preproc", "L2_graph", "L3_recordwise", "ALL"]].mean()
s = df[["LF", "L1_preproc", "L2_graph", "L3_recordwise", "ALL"]].std()
print("\n=== Leakage inflation (AUC mean +/- std; delta vs LF) ===")
for c in ["LF", "L1_preproc", "L2_graph", "L3_recordwise", "ALL"]:
    print(f"  {c:<13} {m[c]:.3f} +/- {s[c]:.3f}   (dAUC = {m[c]-m['LF']:+.3f})")
print("Saved exp3_leakage.csv  total %.0fs" % (time.time()-t0))
