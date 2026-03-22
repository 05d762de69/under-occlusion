from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .modeling import ModelBundle


@torch.no_grad()
def run_model(bundle: ModelBundle, x: torch.Tensor) -> Dict:
    logits = bundle.torch_model(x)
    probs = F.softmax(logits, dim=1)

    pred_class = int(torch.argmax(logits, dim=1).item())
    pred_logit = float(logits[0, pred_class].item())
    pred_prob = float(probs[0, pred_class].item())

    return {
        "logits": logits,
        "probs": probs,
        "pred_class": pred_class,
        "pred_logit": pred_logit,
        "pred_prob": pred_prob,
    }