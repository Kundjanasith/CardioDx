from __future__ import annotations
import numpy as np

def integrated_gradients_torch(model, x, target_index: int, steps: int = 32, baseline=None):
    """Minimal Integrated Gradients for torch ECG models. x shape: (1, leads, samples)."""
    import torch
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(x)
    scaled = [baseline + (float(i) / steps) * (x - baseline) for i in range(1, steps + 1)]
    grads = []
    for sx in scaled:
        sx = sx.detach().clone().requires_grad_(True)
        out = model(sx)
        score = out[:, target_index].sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        grads.append(sx.grad.detach())
    avg_grad = torch.stack(grads).mean(dim=0)
    attr = (x - baseline) * avg_grad
    return attr.detach().cpu().numpy()

def summarize_attribution_by_lead(attr: np.ndarray, leads: list[str]) -> dict:
    a = np.asarray(attr)
    if a.ndim == 3:
        a = a[0]
    scores = np.mean(np.abs(a), axis=1)
    total = float(scores.sum() + 1e-9)
    return {lead: float(scores[i] / total) for i, lead in enumerate(leads[: len(scores)])}
