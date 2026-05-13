"""3-layer MLP probe with patient-grouped CV and early stopping.

Architecture: ``Linear(C, hidden) → GELU → Dropout → Linear(hidden, hidden) →
GELU → Dropout → Linear(hidden, 1)``. Trained with AdamW and early stopping on
an inner train/val split inside each outer patient-grouped fold. The reported
metric is the mean held-out R² across outer folds; the returned model is refit
on the full data with the same recipe.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import Tensor, nn

from experiments.E2._lib.probes.cv import patient_grouped_kfold_indices
from experiments.E2._lib.probes.metrics import r2_score

logger = logging.getLogger(__name__)


class MLPProbe(nn.Module):
    """``C -> hidden -> hidden -> 1`` regressor."""

    def __init__(self, in_features: int, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


def _train_with_early_stop(
    z_tr: np.ndarray,
    y_tr: np.ndarray,
    z_va: np.ndarray,
    y_va: np.ndarray,
    *,
    hidden: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[MLPProbe, float]:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    torch.manual_seed(int(seed))

    in_features = int(z_tr.shape[1])
    model = MLPProbe(in_features=in_features, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    z_tr_t = torch.as_tensor(z_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.float32, device=device)
    z_va_t = torch.as_tensor(z_va, dtype=torch.float32, device=device)
    y_va_t = torch.as_tensor(y_va, dtype=torch.float32, device=device)

    best_val = np.inf
    best_state: dict[str, Tensor] | None = None
    bad = 0
    n = z_tr_t.shape[0]

    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(n, generator=g).to(device)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            pred = model(z_tr_t[idx])
            loss = loss_fn(pred, y_tr_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(z_va_t), y_va_t).item())
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                logger.debug("early stop at epoch %d (val_loss=%.4f)", epoch, val_loss)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def _predict(model: MLPProbe, z: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        z_t = torch.as_tensor(z, dtype=torch.float32, device=device)
        return model(z_t).detach().cpu().numpy()


def fit_mlp_probe(
    z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    hidden: int = 256,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 64,
    n_splits: int = 5,
    inner_val_frac: float = 0.2,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[MLPProbe, float]:
    """Fit an MLP probe; return ``(model_refit_on_all, mean_outer_R²)``.

    Outer loop: patient-grouped K-fold producing held-out test sets. Inside each
    outer fold an inner random split (without grouping — already inside one
    patient block) provides the early-stopping validation set.
    """
    if z.ndim != 2 or y.ndim != 1 or groups.ndim != 1:
        raise ValueError("z must be (N,C); y and groups must be (N,)")
    dev = torch.device(device)
    rng = np.random.default_rng(seed)

    outer_scores: list[float] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(
        patient_grouped_kfold_indices(groups, n_splits=n_splits)
    ):
        z_train, y_train = z[tr_idx], y[tr_idx]
        z_test, y_test = z[te_idx], y[te_idx]
        n_tr = z_train.shape[0]
        n_va = max(1, int(round(n_tr * inner_val_frac)))
        perm = rng.permutation(n_tr)
        va_idx = perm[:n_va]
        tr_only = perm[n_va:]
        model, _ = _train_with_early_stop(
            z_train[tr_only],
            y_train[tr_only],
            z_train[va_idx],
            y_train[va_idx],
            hidden=hidden,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            device=dev,
            seed=seed + fold_idx,
        )
        score = r2_score(_predict(model, z_test, dev), y_test)
        outer_scores.append(score)

    cv_score = float(np.mean(outer_scores))

    # Refit on ALL data with the same recipe; carve out an internal val split
    # again for early stopping. This is the "production" model returned.
    n = z.shape[0]
    n_va = max(1, int(round(n * inner_val_frac)))
    perm = rng.permutation(n)
    final_model, _ = _train_with_early_stop(
        z[perm[n_va:]],
        y[perm[n_va:]],
        z[perm[:n_va]],
        y[perm[:n_va]],
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        device=dev,
        seed=seed + n_splits,
    )
    return final_model, cv_score
