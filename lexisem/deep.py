"""Deep-learning path: LSTM sentiment classifier (paper section 3.4.2).

Reproduces the paper's core deep-learning claim: replacing zero padding with
semantically-rich sentiment vectors ("semantic padding") improves an LSTM
classifier. We train the *same* 4-layer LSTM twice — once with ordinary zero
padding, once with semantic padding — and compare.

Requires PyTorch. Uses the GPU automatically when available.
"""
from __future__ import annotations
from typing import List, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    _TORCH = True
except Exception:  # torch optional
    _TORCH = False

from .features import semantic_padding
from .lexicon import Lexicon


def _zero_padding(tokens, kv, max_len, dim, side="post"):
    def vec(w):
        try:
            return kv[w]
        except KeyError:
            return np.zeros(dim, dtype=np.float32)
    real = [vec(t) for t in tokens[:max_len]]
    npad = max_len - len(real)
    pad = [np.zeros(dim, dtype=np.float32)] * npad
    rows = (pad + real) if side == "pre" else (real + pad)   # pre: real at END
    return np.vstack(rows[:max_len]).astype(np.float32)


if _TORCH:

    class SeqDataset(Dataset):
        """Builds (max_len, dim) embedding matrices with the chosen padding."""
        def __init__(self, docs_tokens: List[List[str]], labels,
                     kv, lex: Lexicon, max_len=300, padding="semantic"):
            self.docs = docs_tokens
            self.labels = labels
            self.kv = kv
            self.lex = lex
            self.max_len = max_len
            self.dim = kv.vector_size
            self.padding = padding

        def __len__(self):
            return len(self.docs)

        def __getitem__(self, i):
            toks = self.docs[i]
            if self.padding == "semantic":
                mat = semantic_padding(toks, self.kv, self.lex,
                                       max_len=self.max_len, vector_size=self.dim)
            else:
                mat = _zero_padding(toks, self.kv, self.max_len, self.dim)
            return torch.from_numpy(mat), torch.tensor(self.labels[i], dtype=torch.long)

    class LSTMClassifier(nn.Module):
        def __init__(self, dim, hidden=128, num_layers=4, dropout=0.2, n_classes=2):
            super().__init__()
            self.lstm = nn.LSTM(dim, hidden, num_layers=num_layers,
                                batch_first=True, dropout=dropout)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )

        def forward(self, x, mask=None):
            out, _ = self.lstm(x)               # (B, T, H)
            if mask is None:
                pooled = out[:, -1, :]          # last time-step (legacy path)
            else:
                m = mask.unsqueeze(-1).to(out.dtype)          # (B, T, 1)
                denom = m.sum(dim=1).clamp(min=1.0)           # (B, 1)
                pooled = (out * m).sum(dim=1) / denom         # masked mean
            return self.head(pooled)

    def train_and_eval(docs_tokens, labels, kv, lex, *, padding="semantic",
                       max_len=300, epochs=6, batch_size=32, lr=1e-3,
                       val_split=0.2, seed=42, verbose=True) -> dict:
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        n = len(docs_tokens)
        idx = np.random.permutation(n)
        cut = int(n * (1 - val_split))
        tr_idx, te_idx = idx[:cut], idx[cut:]

        def subset(ix):
            return [docs_tokens[i] for i in ix], np.asarray(labels)[ix]

        tr_docs, tr_y = subset(tr_idx)
        te_docs, te_y = subset(te_idx)

        tr = SeqDataset(tr_docs, tr_y, kv, lex, max_len, padding)
        te = SeqDataset(te_docs, te_y, kv, lex, max_len, padding)
        tr_dl = DataLoader(tr, batch_size=batch_size, shuffle=True)
        te_dl = DataLoader(te, batch_size=batch_size)

        model = LSTMClassifier(kv.vector_size).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()

        for ep in range(epochs):
            model.train()
            tot = 0.0
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(yb)
            if verbose:
                print(f"    [{padding}] epoch {ep+1}/{epochs} loss={tot/len(tr):.4f}")

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in te_dl:
                xb = xb.to(device)
                p = model(xb).argmax(1).cpu().numpy()
                preds.extend(p.tolist())
                gts.extend(yb.numpy().tolist())

        from sklearn.metrics import precision_recall_fscore_support
        p, r, f, _ = precision_recall_fscore_support(
            gts, preds, labels=[1, 0], zero_division=0)
        return {"PP": p[0], "RP": r[0], "F1_P": f[0],
                "PN": p[1], "RN": r[1], "F1_N": f[1]}

    # ------------------------------------------------------------------ #
    # Paired k-fold comparison (matches the classical protocol).
    #
    # Semantic padding is deterministic, so each doc's (max_len, dim) matrix
    # is built ONCE per condition and reused across folds. Within a fold both
    # conditions get the same fold split, the same weight init, the same
    # minibatch order and the same dropout-mask sequence (all driven by one
    # per-fold seed) — so the ONLY difference is zero- vs semantic-padded
    # feature values. macro-F1 = mean(F1_pos, F1_neg), as in the classical run.
    #
    # READOUT: masked mean-pool over LSTM outputs (not the last time-step).
    #   * zero padding   -> pool over REAL token positions only (pads ignored)
    #   * semantic padding-> pool over ALL positions (its pads are content)
    # This gives the zero baseline a fair chance to train (the last-time-step
    # readout collapsed it to one class on short text) and isolates the actual
    # intervention: whether the sentiment-filled pad region adds signal.
    #
    # STABILITY: lower LR + gradient clipping. A fold whose eval predictions are
    # all one class is a dead-start "collapse"; we retry once with a perturbed
    # seed and, if it still collapses, FLAG it rather than silently averaging.
    # ------------------------------------------------------------------ #
    def _build_matrices(docs_tokens, kv, lex, max_len, padding, side="post"):
        """Return (X, lengths). lengths = #real tokens (capped at max_len)."""
        dim = kv.vector_size
        out = np.empty((len(docs_tokens), max_len, dim), dtype=np.float32)
        lengths = np.empty(len(docs_tokens), dtype=np.int64)
        for i, toks in enumerate(docs_tokens):
            lengths[i] = min(len(toks), max_len)
            if padding == "semantic":
                out[i] = semantic_padding(toks, kv, lex, max_len=max_len,
                                          vector_size=dim, side=side)
            else:
                out[i] = _zero_padding(toks, kv, max_len, dim, side=side)
        return out, lengths

    def _mask_from_lengths(lengths, max_len, full):
        """Bool mask (N, max_len). full=True -> pool over everything (semantic)."""
        if full:
            return np.ones((len(lengths), max_len), dtype=np.float32)
        ar = np.arange(max_len)[None, :]
        return (ar < lengths[:, None]).astype(np.float32)

    def _train_eval_on(X_tr, M_tr, y_tr, X_te, M_te, y_te, dim, *, epochs,
                       batch_size, lr, seed, device, clip=5.0):
        from sklearn.metrics import f1_score
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = LSTMClassifier(dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        use_mask = M_tr is not None   # None -> last-time-step readout

        tensors = [torch.from_numpy(X_tr)]
        if use_mask:
            tensors.append(torch.from_numpy(M_tr))
        tensors.append(torch.tensor(y_tr, dtype=torch.long))
        ds = torch.utils.data.TensorDataset(*tensors)
        g = torch.Generator()
        g.manual_seed(seed)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)

        for _ in range(epochs):
            model.train()
            for batch in dl:
                if use_mask:
                    xb, mb, yb = batch
                    mb = mb.to(device)
                else:
                    xb, yb = batch
                    mb = None
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(model(xb, mb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()

        model.eval()
        Xte = torch.from_numpy(X_te).to(device)
        Mte = torch.from_numpy(M_te).to(device) if use_mask else None
        preds = []
        with torch.no_grad():
            for s in range(0, len(Xte), 256):
                mb = Mte[s:s + 256] if use_mask else None
                preds.append(model(Xte[s:s + 256], mb).argmax(1).cpu())
        pred = torch.cat(preds).numpy()
        collapsed = len(np.unique(pred)) == 1        # all one class -> dead
        f1p = f1_score(y_te, pred, pos_label=1, zero_division=0)
        f1n = f1_score(y_te, pred, pos_label=0, zero_division=0)
        return 0.5 * (f1p + f1n), collapsed

    def _run_cond(X, M, tr, te, labels, dim, *, epochs, batch_size, lr, seed,
                  device):
        """Train/eval one condition; retry once on collapse with a new seed."""
        Mtr = M[tr] if M is not None else None
        Mte = M[te] if M is not None else None
        f1, col = _train_eval_on(X[tr], Mtr, labels[tr], X[te], Mte,
                                 labels[te], dim, epochs=epochs,
                                 batch_size=batch_size, lr=lr, seed=seed,
                                 device=device)
        if col:  # dead-start: one retry with a perturbed seed
            f1_r, col_r = _train_eval_on(X[tr], Mtr, labels[tr], X[te], Mte,
                                         labels[te], dim, epochs=epochs,
                                         batch_size=batch_size, lr=lr,
                                         seed=seed + 1000, device=device)
            if not col_r or f1_r > f1:
                f1, col = f1_r, col_r
        return f1, col

    def kfold_compare(docs_tokens, labels, kv, lex, *, max_len=300, epochs=10,
                      batch_size=32, lr=5e-4, n_splits=5, seed=42,
                      readout="meanpool", verbose=True):
        """readout='meanpool': post-pad + masked mean-pool (zero over real
        tokens, semantic over all slots). readout='prelast': pre-pad + last
        time-step (real token last; pads consumed sequentially first)."""
        from sklearn.model_selection import StratifiedKFold
        labels = np.asarray(labels)
        dim = kv.vector_size
        if verbose:
            print(f"      precomputing padding matrices [readout={readout}] "
                  f"(N={len(docs_tokens)}, max_len={max_len}, dim={dim}) ...")
        if readout == "prelast":
            X_zero, lengths = _build_matrices(docs_tokens, kv, lex, max_len,
                                              "zero", side="pre")
            X_sem, _ = _build_matrices(docs_tokens, kv, lex, max_len,
                                       "semantic", side="pre")
            M_zero = M_sem = None                       # last-time-step readout
        else:
            X_zero, lengths = _build_matrices(docs_tokens, kv, lex, max_len,
                                              "zero", side="post")
            X_sem, _ = _build_matrices(docs_tokens, kv, lex, max_len,
                                       "semantic", side="post")
            M_zero = _mask_from_lengths(lengths, max_len, full=False)  # real toks
            M_sem = _mask_from_lengths(lengths, max_len, full=True)    # all slots

        device = "cuda" if torch.cuda.is_available() else "cpu"
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        zero_f, sem_f, zero_col, sem_col = [], [], [], []
        for fold, (tr, te) in enumerate(skf.split(X_zero, labels)):
            fs = seed + fold
            z, zc = _run_cond(X_zero, M_zero, tr, te, labels, dim, epochs=epochs,
                              batch_size=batch_size, lr=lr, seed=fs, device=device)
            s, sc = _run_cond(X_sem, M_sem, tr, te, labels, dim, epochs=epochs,
                              batch_size=batch_size, lr=lr, seed=fs, device=device)
            zero_f.append(z); sem_f.append(s)
            zero_col.append(bool(zc)); sem_col.append(bool(sc))
            if verbose:
                flag = ""
                if zc or sc:
                    flag = "  [COLLAPSE" + \
                        (" zero" if zc else "") + (" semantic" if sc else "") + "]"
                print(f"      fold {fold+1}/{n_splits}: "
                      f"zero={z:.4f}  semantic={s:.4f}  Δ={s-z:+.4f}{flag}")
        return {"zero_macro_f1": zero_f, "semantic_macro_f1": sem_f,
                "zero_collapsed": zero_col, "semantic_collapsed": sem_col}

else:
    def train_and_eval(*a, **k):  # pragma: no cover
        raise ImportError("PyTorch is required for the LSTM path. "
                          "Install with: pip install torch")

    def kfold_compare(*a, **k):  # pragma: no cover
        raise ImportError("PyTorch is required for the LSTM path. "
                          "Install with: pip install torch")
