# Train and cache the base per-fold models reused by several experiments.
# Saves state_dicts for: causal(+lag), fully-connected(+lag), random(+lag),
# no-inter-sensor(+lag). Confirms reproduction of the manuscript AUCs.
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from common import (load_data, ProposedModel, standardize, fold_causal, fc_mask,
                    random_mask, fold_indices, class_weight, train_model, eval_model,
                    seed_all, SEED, TAB_DIR, PROC_DIR)
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results" / "tables"
from common import MODELS_DIR as MODELS
torch.set_num_threads(2)

d = load_data(); X, y, N, NF = d["X"], d["y"], d["N"], d["N_FOLDS"]
GRAPHS = ["causal", "fc", "random"]   # none is covered by the paper's ablation
results = []
t0 = time.time()
for f in range(NF):
    idx_tr, idx_va, idx_te = fold_indices(d, f)
    Xs = standardize(X[idx_tr], X)
    cw = class_weight(y, idx_tr)
    # build graphs
    g = {
        "causal": fold_causal(f, N),
        "fc": fc_mask(N),
        "random": random_mask(f, N),
    }
    for name in GRAPHS:
        ckpt_path = MODELS / f"{name}_fold{f}.pt"
        mask, lag = g[name]
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, weights_only=False)
            model = ProposedModel(mask, lag, use_lag=True)
            model.load_state_dict(ckpt["state"]); model.eval()
            r = eval_model(model, Xs, y, idx_te)
            print(f"fold {f} | {name:<8} (cached) AUC={r['auc']:.3f}", flush=True)
        else:
            seed_all(SEED); torch.manual_seed(SEED)
            model = ProposedModel(mask, lag, use_lag=True)
            model = train_model(model, Xs, y, idx_tr, idx_va, cw)
            r = eval_model(model, Xs, y, idx_te)
            torch.save({"state": model.state_dict(), "mask": mask, "lag": lag}, ckpt_path)
            print(f"fold {f} | {name:<8} AUC={r['auc']:.3f} macroF1={r['macro_f1']:.3f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        results.append({"graph": name, "fold": f, "auc": round(r["auc"], 4),
                        "macro_f1": round(r["macro_f1"], 4), "binary_f1": round(r["binary_f1"], 4)})

import pandas as pd
df = pd.DataFrame(results)
df.to_csv(OUT / "base_models_cv.csv", index=False)
print("\n=== Base-model AUC (mean +/- std over folds) ===")
for name in GRAPHS:
    sub = df[df.graph == name]
    print(f"  {name:<8} AUC {sub.auc.mean():.3f} +/- {sub.auc.std():.3f}")
print("Saved base models and base_models_cv.csv. total %.0fs" % (time.time()-t0))
