"""Membership probe: can a linear probe on mean-pooled ESM-2 embeddings tell family
F from background? Reported as held-out AUROC. Secondary acquisition readout — more
robust to perplexity noise than PPL alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class ProbeResult:
    auroc: float
    n_train: int
    n_test: int
    n_features: int

    def as_dict(self) -> dict:
        return {"membership_auroc": self.auroc, "n_train": self.n_train, "n_test": self.n_test}


def embed_sequences(
    model,
    tokenizer,
    sequences: Sequence[str],
    device: str = "cpu",
    batch_size: int = 32,
    max_length: int = 1024,
):
    """Mean-pool the last hidden state over non-special, non-pad tokens. Returns a
    numpy array [n_sequences, hidden_dim]."""
    import numpy as np
    import torch

    was_training = model.training
    model.eval()
    feats: List = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            chunk = list(sequences[start : start + batch_size])
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            special = enc["special_tokens_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
            hidden = out.hidden_states[-1]  # [b, L, H]
            pool_mask = (attn.bool() & (special == 0)).unsqueeze(-1).to(hidden.dtype)  # [b, L, 1]
            summed = (hidden * pool_mask).sum(dim=1)
            counts = pool_mask.sum(dim=1).clamp(min=1.0)
            mean = (summed / counts).float().cpu().numpy()
            feats.append(mean)
    if was_training:
        model.train()
    return np.concatenate(feats, axis=0)


def probe_auroc(train_pos, train_neg, test_pos, test_neg, seed: int = 0) -> ProbeResult:
    """Fit a logistic-regression probe on (train_pos vs train_neg) embeddings and
    report AUROC on (test_pos vs test_neg)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    Xtr = np.concatenate([train_pos, train_neg], axis=0)
    ytr = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
    Xte = np.concatenate([test_pos, test_neg], axis=0)
    yte = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(scaler.transform(Xtr), ytr)
    scores = clf.predict_proba(scaler.transform(Xte))[:, 1]
    auroc = float(roc_auc_score(yte, scores))
    return ProbeResult(auroc=auroc, n_train=len(ytr), n_test=len(yte), n_features=Xtr.shape[1])


def membership_auroc(
    model,
    tokenizer,
    family_seqs: Sequence[str],
    background_seqs: Sequence[str],
    device: str = "cpu",
    batch_size: int = 32,
    max_length: int = 1024,
    test_frac: float = 0.5,
    seed: int = 0,
) -> ProbeResult:
    """End-to-end: embed family + background, split, fit probe, report held-out AUROC."""
    from sklearn.model_selection import train_test_split

    pos = embed_sequences(model, tokenizer, family_seqs, device, batch_size, max_length)
    neg = embed_sequences(model, tokenizer, background_seqs, device, batch_size, max_length)
    pos_tr, pos_te = train_test_split(pos, test_size=test_frac, random_state=seed)
    neg_tr, neg_te = train_test_split(neg, test_size=test_frac, random_state=seed)
    return probe_auroc(pos_tr, neg_tr, pos_te, neg_te, seed=seed)
