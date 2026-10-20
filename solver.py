"""Closed-form ridge editing in the retained inputs' orthogonal complement."""

from __future__ import annotations

import torch


@torch.no_grad()
def solve_factors(source, anchor, retain, *, beta=1.0, rank_rtol=1e-6,
                  preserve_rank=None, device="cpu"):
    """B=L@R for the same constrained ridge problem, without a full null basis.

    Thin SVD projects the inputs onto the retained-space complement. The
    smaller of the sample/input Gram systems is solved. This avoids a dense
    4096-dimensional null basis for T5 and supports low-rank weight updates.
    Samples are rows, and the actual module update is (W@L)@R.
    """
    if source.ndim != 2 or source.shape != anchor.shape or source.shape[0] == 0:
        raise ValueError("Source and anchor must have identical nonempty [N,d] shapes.")
    if retain.ndim != 2 or retain.shape[1] != source.shape[1]:
        raise ValueError("Retain inputs must be [M,d] in the same input space.")
    if beta <= 0 or rank_rtol < 0 or (preserve_rank is not None and preserve_rank < 0):
        raise ValueError("Invalid ridge or preservation rank configuration.")
    xt, xa, xp = [x.to(device=device, dtype=torch.float64) for x in (source, anchor, retain)]
    if not all(torch.isfinite(x).all() for x in (xt, xa, xp)):
        raise ValueError("Non-finite module inputs.")
    n, d = xt.shape
    if len(xp):
        _, singular, vh = torch.linalg.svd(xp, full_matrices=False)
        numerical_rank = int((singular > singular[0] * rank_rtol).sum())
        rank = min(numerical_rank, preserve_rank) if preserve_rank is not None else numerical_rank
        u = vh[:rank].T
    else:
        numerical_rank = rank = 0
        u = xt.new_empty((d, 0))
    if rank == d:
        raise ValueError("Retain inputs span all input dimensions; reduce retains or explicitly truncate their rank.")
    projected = xt - (xt @ u) @ u.T
    if projected.norm() <= 1e-12 * xt.norm().clamp_min(1):
        raise ValueError("Target inputs have no usable direction in the preservation null space.")
    if n <= d:
        gram = projected @ projected.T
        gram.diagonal().add_(beta)
        left = (xa - xt).T
        right = torch.linalg.solve(gram, projected)
        system = "sample"
    else:
        gram = projected.T @ projected
        gram.diagonal().add_(beta)
        left = (xa - xt).T @ projected
        right = torch.linalg.solve(gram, torch.eye(d, device=device, dtype=torch.float64))
        system = "input"
    # Remove numerical leakage from the ridge solve without changing its form.
    right = right - (right @ u) @ u.T
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("Non-finite closed-form factors.")
    report = {
        "source_samples": n, "retain_samples": len(xp), "input_dim": d,
        "numerical_retain_rank": numerical_rank, "protected_rank": rank,
        "null_dim": d - rank, "rank_truncated": rank < numerical_rank,
        "beta": beta, "linear_system": system,
        "relative_retain_change": float(
            ((xp @ right.T) @ left.T).norm() / xp.norm().clamp_min(1e-12)
        ),
    }
    return left.cpu(), right.cpu(), report


@torch.no_grad()
def solve_transform(
    source: torch.Tensor,
    anchor: torch.Tensor,
    retain: torch.Tensor,
    *,
    beta: float = 1.0,
    rank_rtol: float = 1e-6,
    preserve_rank: int | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """Return B so each module update is W @ B.

    Input samples are ROWS: source/anchor [N,d], retain [M,d].
    With null basis Z and A=X_T.T@Z, solve
      B=(X_A-X_T).T @ A @ (A.T@A+beta*I)^(-1) @ Z.T.
    This is the original constrained closed form in reduced coordinates.
    All linear algebra uses float64; no explicit inverse is formed.
    """
    if source.ndim != 2 or source.shape != anchor.shape or source.shape[0] == 0:
        raise ValueError("Source and anchor must have identical nonempty [N,d] shapes.")
    if retain.ndim != 2 or retain.shape[1] != source.shape[1]:
        raise ValueError("Retain inputs must be [M,d] in the same input space.")
    if beta <= 0 or rank_rtol < 0:
        raise ValueError("beta must be positive and rank_rtol nonnegative.")
    if preserve_rank is not None and preserve_rank < 0:
        raise ValueError("preserve_rank must be nonnegative.")
    
    xt = source.to(device=device, dtype=torch.float64)
    xa = anchor.to(device=device, dtype=torch.float64)
    xp = retain.to(device=device, dtype=torch.float64)
    if not all(torch.isfinite(x).all() for x in (xt, xa, xp)):
        raise ValueError("Non-finite module inputs.")
    d = xt.shape[1]
    if len(xp):
        u, singular, _ = torch.linalg.svd(xp.T, full_matrices=True)
        cutoff = float(singular[0]) * rank_rtol if len(singular) else 0.0
        numerical_rank = int((singular > cutoff).sum())
        rank = min(numerical_rank, preserve_rank) if preserve_rank is not None else numerical_rank
        z = u[:, rank:]
    else:
        numerical_rank = rank = 0
        z = torch.eye(d, device=device, dtype=torch.float64)
    if z.shape[1] == 0:
        raise ValueError(
            f"Retain inputs span all {d} input dimensions. Reduce --retain-count or "
            "explicitly choose --preserve-rank (this relaxes full retain protection)."
        )
    a = xt @ z  # [N, null_dim]
    if a.norm() <= 1e-12 * xt.norm().clamp_min(1):
        raise ValueError("Target inputs have no usable direction in the preservation null space.")
    
    gram = a.T @ a
    gram.diagonal().add_(beta)
    solved = torch.linalg.solve(gram, a.T)
    transform = (xa - xt).T @ solved.T @ z.T
    if not torch.isfinite(transform).all():
        raise ValueError("The closed-form update is not finite.")
    diagnostics = {
        "source_samples": len(xt),
        "retain_samples": len(xp),
        "input_dim": d,
        "numerical_retain_rank": numerical_rank,
        "protected_rank": rank,
        "null_dim": z.shape[1],
        "rank_truncated": rank < numerical_rank,
        "beta": beta,
        "relative_retain_change": float(
            (xp @ transform.T).norm() / xp.norm().clamp_min(1e-12)
        ),
    }
    return transform.cpu(), diagnostics
