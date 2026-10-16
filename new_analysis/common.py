# ============================================================
# Shared module for novelty-forward experiments.
# Faithfully reuses the proposed model, standardization, fold
# protocol, and graph construction from notebooks 03-07, so new
# results are directly comparable to the manuscript's.
# ============================================================
from pathlib import Path
import numpy as np, pandas as pd, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score

SEED = 42
EPOCHS = 25; LR = 1e-3; WD = 1e-4; BATCH = 256
TAU_MAX = 6

# Repo-relative paths: this file lives in <repo>/new_analysis/
_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = _ROOT / "data" / "processed"
TAB_DIR  = _ROOT / "results" / "tables"
MODELS_DIR = Path(__file__).resolve().parent / "_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def load_data():
    z = np.load(PROC_DIR / "windows.npz", allow_pickle=True)
    d = dict(
        X=z["X"].astype(np.float32), y=z["y"].astype(np.int64), hh=z["hh"],
        sample_fold=z["sample_fold"].astype(int),
        FEAT=list(z["feat_cols"]),
        EM_IDX=list(z["em_idx"]), SM_IDX=list(z["sm_idx"]),
        WINDOW=int(z["window"]),
    )
    d["N"] = len(d["FEAT"]); d["N_FOLDS"] = int(d["sample_fold"].max()) + 1
    return d

# ── Proposed model (verbatim architecture from nb04) ─────────
class ProposedModel(nn.Module):
    def __init__(self, mask, lag, use_lag=True, node_dim=32, heads=4, hid=64, dp=0.3):
        super().__init__()
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.register_buffer("lag", torch.tensor(lag, dtype=torch.float32))
        self.use_lag = use_lag; self.h = heads; self.hd = hid // heads; self.hid = hid
        self.n = mask.shape[0]
        self.temp = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(16, node_dim, 3, padding=1), nn.ReLU(),
                                  nn.AdaptiveAvgPool1d(1))
        self.q = nn.Linear(node_dim, hid); self.k = nn.Linear(node_dim, hid); self.v = nn.Linear(node_dim, hid)
        self.lagw = nn.Parameter(torch.zeros(self.h)) if use_lag else None
        self.proj = nn.Linear(hid, hid); self.drop = nn.Dropout(dp)
        self.clf = nn.Sequential(nn.Linear(hid, 32), nn.ReLU(), nn.Dropout(dp), nn.Linear(32, 2))

    def set_graph(self, mask, lag):
        """Swap the attention graph at inference (for occlusion / personalization)."""
        self.mask.copy_(torch.tensor(mask, dtype=torch.float32))
        self.lag.copy_(torch.tensor(lag, dtype=torch.float32))

    def forward(self, x, return_attn=False):
        B = x.shape[0]; Nn = self.n
        h = self.temp(x.reshape(B * Nn, 1, x.shape[-1])).squeeze(-1).view(B, Nn, -1)
        h = self.drop(h)
        q = self.q(h).view(B, Nn, self.h, self.hd).transpose(1, 2)
        k = self.k(h).view(B, Nn, self.h, self.hd).transpose(1, 2)
        v = self.v(h).view(B, Nn, self.h, self.hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.hd ** 0.5)
        eye = torch.eye(Nn, device=x.device)[None, None]
        allow = ((self.mask[None, None] + eye) > 0).float()
        if self.use_lag:
            att = att + self.lagw.view(1, self.h, 1, 1) * self.lag[None, None]
        att = att.masked_fill(allow == 0, float("-inf"))
        att = torch.nan_to_num(torch.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, Nn, self.hid)
        out = self.drop(F.elu(self.proj(out)))
        logits = self.clf(out.mean(dim=1))
        if return_attn:
            return logits, att            # att: (B, heads, Nn, Nn)
        return logits

def standardize(Xtr_raw, Xall_raw):
    mu = Xtr_raw.mean(axis=(0, 2), keepdims=True)
    sd = Xtr_raw.std(axis=(0, 2), keepdims=True); sd[sd == 0] = 1.0
    return (Xall_raw - mu) / sd

# ── Graph builders ───────────────────────────────────────────
def adj_to_mask_lag(adj, N, idx=None):
    ii = list(range(N)) if idx is None else list(idx)
    n = len(ii); mask = np.zeros((n, n), np.float32); lag = np.zeros((n, n), np.float32)
    for a, i in enumerate(ii):
        for b, j in enumerate(ii):
            taus = [t for t in range(1, adj.shape[2]) if adj[i, j, t] == 1]
            if taus:
                mask[b, a] = 1.0; lag[b, a] = float(min(taus))   # [target, source]
    return mask, lag

def fold_causal(f, N, idx=None):
    adj = np.load(PROC_DIR / f"causal_adj_fold{f}.npy")
    return adj_to_mask_lag(adj, N, idx)

def fc_mask(N):
    m = np.ones((N, N), np.float32); np.fill_diagonal(m, 0.0)
    return m, np.zeros((N, N), np.float32)

def random_mask(f, N, seed=SEED):
    cm, _ = fold_causal(f, N); ne = int(cm.sum())
    r = np.random.default_rng(seed)
    m = np.zeros((N, N), np.float32)
    pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    for c in r.choice(len(pairs), size=min(ne, len(pairs)), replace=False):
        m[pairs[c]] = 1.0
    return m, np.zeros((N, N), np.float32)

# ── Per-household causal links (cached, no PCMCI rerun) ──────
def household_links():
    """dict: HH_ID -> set of (src_idx, dst_idx, lag)."""
    df = pd.read_csv(TAB_DIR / "causal_links_per_household.csv")
    links = {}
    for hh, g in df.groupby("HH_ID"):
        links[hh] = set((int(r.src_idx), int(r.dst_idx), int(r.Lag)) for r in g.itertuples())
    return links

def links_to_adj(link_sets, N, freq_thresh=0.50):
    """Aggregate a list of link-sets into (N,N,tau+1) adjacency at freq>=thresh."""
    cnt = np.zeros((N, N, TAU_MAX + 1))
    for s in link_sets:
        for (i, j, t) in s:
            cnt[i, j, t] += 1
    freq = cnt / max(len(link_sets), 1)
    return (freq >= freq_thresh).astype(np.float32)

# ── Training / evaluation (verbatim protocol from nb04) ──────
def fold_indices(d, f):
    va = (f + 1) % d["N_FOLDS"]
    idx_te = np.where(d["sample_fold"] == f)[0]
    idx_va = np.where(d["sample_fold"] == va)[0]
    idx_tr = np.where((d["sample_fold"] != f) & (d["sample_fold"] != va))[0]
    return idx_tr, idx_va, idx_te

def class_weight(y, idx_tr):
    return torch.tensor([1.0, (y[idx_tr] == 0).sum() / max((y[idx_tr] == 1).sum(), 1)], dtype=torch.float)

def _loader(Xs, y, idx, shuffle):
    return DataLoader(TensorDataset(torch.tensor(Xs[idx]), torch.tensor(y[idx])),
                      batch_size=BATCH, shuffle=shuffle)

def train_model(model, Xs, y, idx_tr, idx_va, cw):
    tr = _loader(Xs, y, idx_tr, True); va = _loader(Xs, y, idx_va, False)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best = -1; best_state = None
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tr:
            loss = crit(model(xb), yb); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step(); model.eval(); P = []; T = []
        with torch.no_grad():
            for xb, yb in va:
                P.extend(model(xb).argmax(1).numpy()); T.extend(yb.numpy())
        vf = f1_score(T, P, average="macro", zero_division=0)
        if vf > best: best = vf; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    return model

def eval_model(model, Xs, y, idx_te, graph=None):
    """Evaluate; optionally swap in a different (mask,lag) graph first."""
    if graph is not None:
        model.set_graph(*graph)
    te = _loader(Xs, y, idx_te, False)
    P = []; T = []; PR = []
    model.eval()
    with torch.no_grad():
        for xb, yb in te:
            lo = model(xb); P.extend(lo.argmax(1).numpy()); T.extend(yb.numpy())
            PR.extend(F.softmax(lo, 1)[:, 1].numpy())
    T = np.array(T); P = np.array(P); PR = np.array(PR)
    return dict(macro_f1=f1_score(T, P, average="macro", zero_division=0),
                binary_f1=f1_score(T, P, average="binary", zero_division=0),
                auc=roc_auc_score(T, PR), T=T, PR=PR)
