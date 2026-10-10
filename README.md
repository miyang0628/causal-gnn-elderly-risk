# Causal-Guided Multimodal Sensor Fusion for Elderly Risk Detection: Interpretable Parity under Leakage-Free Evaluation

> **Anonymous repository for double-blind peer review.**
> Author, institution, and affiliation details are intentionally omitted.
> Submitted to an anonymous journal.

Reproducibility package (code, data, and results) for the manuscript on
**in-home risk detection in single-occupancy elderly households**, fusing
**multimodal structured IoT sensor streams** (environmental and physiological)
under the guidance of **temporal causal discovery** (PCMCI).

This repository re-implements the full study with a corrected, leakage-free
evaluation protocol: baselines trained under an identical fair setting, a
controlled ablation that isolates one factor at a time, and household-level
cross-validation with causal graphs rebuilt inside each training fold.

The pipeline is a sequence of self-contained Jupyter notebooks. Each notebook
holds its entire code in a single cell, begins with an English header comment
naming the notebook, and writes its outputs to `results/`.

---

## Repository structure

```
.
├── data/
│   └── processed/df_new.pkl   # parsed dataset (27 households) — not redistributed
├── notebooks/                 # analysis pipeline (run in order)
├── results/
│   ├── figures/               # figures (.png and .pdf, 600 dpi, grayscale)
│   └── tables/                # result tables (.csv)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Pipeline (notebooks)

Run in numerical order. Each notebook reads cached outputs of the previous steps.

| Notebook | Purpose |
|----------|---------|
| `00_setup_and_config`        | Environment, global paths, shared figure style (grayscale, 600 dpi) |
| `01_data_preparation`        | Load `df_new.pkl`, build the binary risk target and sliding windows, define the household-level split |
| `02_eda`                     | Exploratory analysis: label distribution, lagged cross-correlation, sensor summaries |
| `03_causal_discovery`        | PCMCI+ temporal causal discovery on **training households only** (per fold) |
| `04_proposed_model`          | Proposed causal-guided graph fusion model |
| `05_baselines`               | Baselines under an identical, fair protocol (same split, scaling, class weighting) |
| `06_ablation`                | Controlled ablation: fixed architecture; vary topology / lag / modality one at a time |
| `07_crossval_generalization` | Group K-fold / leave-one-household-out with mean ± std and significance tests |
| `08_figures_tables`          | Final figures and tables for the manuscript |

---

## Figure conventions

A single publication style, applied everywhere:

- **seaborn**, **grayscale** palette,
- **600 dpi**,
- saved as **both `.png` and `.pdf`** to `results/figures/`,
- **no in-figure captions** (captions live in the manuscript).

---

## Reproducibility

- All random seeds are fixed (`SEED = 42`).
- The **causal graph is reconstructed within each training fold only**, so
  held-out households never influence the model structure.
- Baselines and the proposed model share an identical split, preprocessing, and
  class weighting, so comparisons are like-for-like.
- Software versions are pinned in `requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab      # run notebooks/ in numerical order
```

---

## Data

See [`data/README.md`](data/README.md). The dataset is not redistributed here.

---

## License

Code released under the MIT License. The dataset retains its original license.
