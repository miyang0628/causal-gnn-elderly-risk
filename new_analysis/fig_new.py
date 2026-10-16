import os
# Generate grayscale, 600-dpi figures and a consolidated summary for the new
# experiments, matching the manuscript's figure conventions.
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "results" / "tables"
FIG = Path(__file__).resolve().parent.parent / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
                     "font.family": "DejaVu Sans", "axes.unicode_minus": False})
GC = {"d": "#222222", "m": "#777777", "l": "#BBBBBB"}
def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

summary = {}

# ── Fig A: faithfulness curve ────────────────────────────────
f1 = pd.read_csv(OUT / "exp1_faithfulness.csv")
g = f1.groupby("frac_kept")
fr = sorted(f1.frac_kept.unique())
def ms(col): return [g.get_group(x)[col].mean() for x in fr], [g.get_group(x)[col].std() for x in fr]
ys, ye = ms("auc_keep_stable"); yr, yre = ms("auc_keep_random"); yl, yle = ms("auc_keep_least")
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.errorbar(fr, ys, yerr=ye, marker="o", color=GC["d"], lw=1.4, capsize=3, label="keep stable-first")
ax.errorbar(fr, yr, yerr=yre, marker="s", color=GC["m"], lw=1.4, capsize=3, label="keep random-first")
ax.errorbar(fr, yl, yerr=yle, marker="^", color=GC["l"], lw=1.4, capsize=3, label="keep least-stable-first")
ax.axhline(0.5, color="black", ls="--", lw=0.8)
ax.set_xlabel("Fraction of causal edges retained"); ax.set_ylabel("AUC-ROC")
ax.legend(fontsize=8); save(fig, "fig_faithfulness")
summary["faithfulness"] = {"frac": fr, "stable": [round(v,3) for v in ys],
                           "random": [round(v,3) for v in yr], "least": [round(v,3) for v in yl]}

# ── Fig B: leakage inflation ─────────────────────────────────
l = pd.read_csv(OUT / "exp3_leakage.csv")
cols = ["LF", "L1_preproc", "L2_graph", "L3_recordwise", "ALL"]
labels = ["Leakage-\nfree", "Preproc\nleak", "Graph\nleak", "Record-wise\nsplit", "All\ncombined"]
m = [l[c].mean() for c in cols]; s = [l[c].std() for c in cols]
fig, ax = plt.subplots(figsize=(5.6, 3.6))
bars = ax.bar(range(len(cols)), m, yerr=s, capsize=3,
              color=[GC["d"]] + [GC["m"]]*3 + [GC["l"]], edgecolor="black", lw=0.6)
ax.axhline(m[0], color=GC["d"], ls=":", lw=1.1)
ax.axhline(0.5, color="black", ls="--", lw=0.8)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("AUC-ROC"); ax.set_ylim(0, 1.0)
for i, v in enumerate(m):
    ax.text(i, v + s[i] + 0.02, f"{v:.2f}", ha="center", fontsize=8)
save(fig, "fig_leakage")
summary["leakage"] = {c: {"mean": round(l[c].mean(),3), "std": round(l[c].std(),3),
                          "dAUC_vs_LF": round(l[c].mean()-l["LF"].mean(),3)} for c in cols}

# ── Fig C: personalization ───────────────────────────────────
p = pd.read_csv(OUT / "exp4_personalized.csv")
cols4 = ["auc_global", "auc_adaptive", "auc_oracle"]
labs4 = ["Global\n(proposed)", "Adaptive\n(deployable)", "Oracle\n(upper bound)"]
mm = [p[c].dropna().mean() for c in cols4]; ss = [p[c].dropna().std() for c in cols4]
fig, ax = plt.subplots(figsize=(4.8, 3.6))
ax.bar(range(3), mm, yerr=ss, capsize=3, color=[GC["m"], GC["d"], GC["l"]], edgecolor="black", lw=0.6)
ax.axhline(0.5, color="black", ls="--", lw=0.8)
ax.set_xticks(range(3)); ax.set_xticklabels(labs4, fontsize=8)
ax.set_ylabel("Per-household AUC-ROC"); ax.set_ylim(0, 1.0)
save(fig, "fig_personalization")
summary["personalization"] = {c: {"mean": round(p[c].dropna().mean(),3),
                                  "std": round(p[c].dropna().std(),3),
                                  "n": int(p[c].dropna().shape[0])} for c in cols4}

# ── merge exp2 json ──────────────────────────────────────────
summary["interpretability"] = json.load(open(OUT / "exp2_interpret.json"))
summary["base_models"] = pd.read_csv(OUT / "base_models_cv.csv").groupby("graph").auc.agg(["mean","std"]).round(4).to_dict("index")

json.dump(summary, open(OUT / "ALL_RESULTS_SUMMARY.json", "w"), indent=2, default=str)
print(json.dumps(summary, indent=2, default=str))
print("\nFigures ->", FIG)
