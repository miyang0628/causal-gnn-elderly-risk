# Separating the Predictive and Interpretive Value of Causal Structure in Multimodal Sensor Fusion for Elderly Risk Detection

Reproducibility package (code, data pointers, and results) for the manuscript on
**in-home risk detection in single-occupancy elderly households**, fusing
**multimodal structured IoT sensor streams** (continuous physiological signals —
heart rate, breathing rate, stress index — together with ambient/environmental
channels) under the guidance of **temporal causal discovery** (PCMCI).

**Authors:** Hyunjeong Kang (Kookmin University, Seoul) and Munil Yang
(Institute for Industrial Policy Studies, Seoul; corresponding author,
miyang@ips.or.kr).
**Target venue:** *Biomedical Signal Processing and Control* (Elsevier).
**Repository:** https://github.com/miyang0628/causal-gnn-elderly-risk

The study separates two questions that the sensor-fusion literature usually fuses:
whether discovered causal structure improves **prediction**, and whether it adds
verifiable **interpretive** value. We measure each directly under a leakage-free,
household-level cross-validation protocol, and validate the protocol itself with a
leakage-injection stress test.

---

## Headline results

| Axis | Finding | Number |
|------|---------|--------|
| Predictive | CausalFusion vs. strong correlation baselines | AUC **0.709 ± 0.123** (statistical parity) |
| Predictive | Controlled ablation: causal vs. fully-connected / random / edge-free | all within **1 std** |
| Predictive | Personalisation (adaptive / oracle graph) | **no gain** (p = 0.08 / 0.02) |
| Interpretive | Cross-subject edge stability (causal vs. random) | Jaccard **0.77 vs. 0.45** |
| Interpretive | Edge faithfulness (keep stable-first vs. random) | e.g. 0.717 vs. 0.696 at 75% edges |
| Interpretive | Does unconstrained attention recover the causal graph? | AUROC **0.56** (no) |
| Interpretive | CO₂–TVOC directional resolution | **not resolved** (both directions, every fold) |
| Robustness | Record-wise split leakage | collapses cross-household std **0.126 → 0.009** |

All numbers are reproduced deterministically (`SEED = 42`). The raw per-fold tables
are in [`results/tables/`](results/tables/) and a machine-readable summary in
[`results/tables/ALL_RESULTS_SUMMARY.json`](results/tables/ALL_RESULTS_SUMMARY.json).

---

## Repository structure

```
.
├── data/
│   └── processed/              # windows.npz, per-fold causal adjacencies, folds
├── notebooks/                  # original pipeline (run in order) — notebooks 00–08
├── new_analysis/               # novelty-forward experiments (this revision)
│   ├── common.py               # shared model + protocol (imported by the scripts)
│   ├── exp_train_base.py       # train & cache base models (causal / FC / random)
│   ├── exp1_faithfulness.py    # edge-faithfulness (occlusion) test
│   ├── exp2_interpret.py       # edge stability, directionality, attention agreement
│   ├── exp3_leakage.py         # leakage-injection stress test
│   ├── exp4_personalized.py    # adaptive / oracle personalised graphs
│   ├── fig_new.py              # figures + ALL_RESULTS_SUMMARY.json
│   └── run_all.py              # runs the five stages in order
├── results/
│   ├── figures/                # figures (.png and .pdf, 600 dpi)
│   └── tables/                 # result tables (.csv) and summary (.json)
├── requirements.txt
└── README.md
```

---

## Pipeline

### Original study (`notebooks/`, run in order)

| Notebook | Purpose |
|----------|---------|
| `00_setup_and_config`        | Environment, global paths, shared figure style |
| `01_data_preparation`        | Build the binary risk target, sliding windows, household split |
| `02_eda`                     | Label distribution, lagged cross-correlation, sensor summaries |
| `03_causal_discovery`        | PCMCI temporal causal discovery on **training households only** (per fold) |
| `04_proposed_model`          | Proposed causal-guided graph-attention model (CausalFusion) |
| `05_baselines`               | Baselines under an identical, fair protocol |
| `06_ablation`                | Controlled ablation: fixed architecture; vary topology / lag / modality |
| `07_crossval_generalization` | Group 5-fold aggregation, significance tests, per-household AUC |
| `08_figures_tables`          | Manuscript figures and tables |

### Novelty-forward experiments (`new_analysis/`)

These add the predictive-vs-interpretive separation that frames the revised paper.
They reuse the cached `data/processed/` artifacts and the exact model/protocol of
notebook 04, so results are directly comparable.

```bash
cd new_analysis
python run_all.py          # runs the five stages below in order, or run individually:

python exp_train_base.py   # trains & caches base models to new_analysis/_models/
python exp1_faithfulness.py
python exp2_interpret.py
python exp3_leakage.py
python exp4_personalized.py
python fig_new.py          # writes figures + results/tables/ALL_RESULTS_SUMMARY.json
```

| Script | Produces |
|--------|----------|
| `exp_train_base.py`    | `results/tables/base_models_cv.csv`; cached models in `new_analysis/_models/` |
| `exp1_faithfulness.py` | `results/tables/exp1_faithfulness.csv` |
| `exp2_interpret.py`    | `results/tables/exp2_interpret.json`, `exp2_directionality.csv` |
| `exp3_leakage.py`      | `results/tables/exp3_leakage.csv` |
| `exp4_personalized.py` | `results/tables/exp4_personalized.csv` |
| `fig_new.py`           | `results/figures/fig_{faithfulness,leakage,personalization}.{png,pdf}`, `ALL_RESULTS_SUMMARY.json` |

> `exp1`, `exp2`, and `exp4` load the base-model checkpoints, so run
> `exp_train_base.py` (or `run_all.py`) first. Training is CPU-friendly
> (~1–2 min per model; 2 threads).

---

## Reproducibility

- All random seeds are fixed (`SEED = 42`); results reproduce deterministically.
- The **causal graph and all preprocessing are rebuilt within each training fold
  only**, so held-out households never influence model structure, scaling, or
  class weights.
- Baselines, the proposed model, and every ablation share an identical split,
  preprocessing, and class weighting — comparisons are like-for-like.
- Software versions are pinned in `requirements.txt`. The `new_analysis/` scripts
  need only `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`, and `matplotlib`
  (not `torch-geometric` or `tigramite`, since they reuse cached graphs).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab                 # notebooks/ in order, then:
python new_analysis/run_all.py
```

---

## Data

See [`data/README.md`](data/README.md). The source dataset is publicly available
from its original provider and is not redistributed here; the processed arrays
needed to reproduce the analyses are under `data/processed/`.

---

## License

Code released under the MIT License. The dataset retains its original license.
