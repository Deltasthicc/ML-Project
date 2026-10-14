"""Training objectives.

    L_total = L_Dice + lambda_1 * L_BCE + lambda_2 * L_topo + lambda_3 * L_cons          (paper, Sec. IV-B)

* ``L_Dice`` / ``L_BCE``     region and pixel terms, computed inside the FOV only.
* ``L_topo``                 selectable: soft-clDice (Shit et al., CVPR'21), Skeleton Recall (Kirchhoff et al.,
                             ECCV'24, binary form) or both, plus the auxiliary skeleton-head loss.
* ``L_cons``                 consistency between the prediction on a clean view and on its style/structure-perturbed
                             view of the same patch (DG stage).

All functions take *probabilities / logits and targets of shape (B, 1, H, W)* and a ``fov`` mask of the same shape.
They are written from the descriptions in the cited papers (not copied from the authors' repositories).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

SMOOTH = 1.0


# ---------------------------------------------------------------------------------- soft morphology
def soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Differentiable 3x3 cross erosion = min-pool in horizontal and vertical direction."""
    p1 = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(p1, p2)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, 3, stride=1, padding=1)


def soft_open(x: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(x))


def soft_skeleton(x: torch.Tensor, iters: int) -> torch.Tensor:
    """Differentiable morphological skeleton (iterated soft opening residuals, as in clDice, Sec. 3.2)."""
    skel = F.relu(x - soft_open(x))
    for _ in range(iters):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


# ---------------------------------------------------------------------------------------- base terms
def dice_loss(prob: torch.Tensor, target: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
    p, t = prob * fov, target * fov
    inter = (p * t).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + SMOOTH) / (den + SMOOTH)).mean()


def bce_loss(logits: torch.Tensor, target: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
    l = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (l * fov).sum() / fov.sum().clamp(min=1.0)


# ------------------------------------------------------------------------------------- topology terms
def soft_cldice_loss(prob: torch.Tensor, target: torch.Tensor, fov: torch.Tensor, target_skel: torch.Tensor | None,
                     iters: int = 10) -> torch.Tensor:
    """1 - clDice, with a differentiable skeleton of the prediction and the exact skeleton of the label."""
    p, t = prob * fov, target * fov
    sp = soft_skeleton(p, iters) * fov
    st = (target_skel * fov) if target_skel is not None else soft_skeleton(t, iters) * fov
    tprec = ((sp * t).sum(dim=(1, 2, 3)) + SMOOTH) / (sp.sum(dim=(1, 2, 3)) + SMOOTH)   # topology precision
    tsens = ((st * p).sum(dim=(1, 2, 3)) + SMOOTH) / (st.sum(dim=(1, 2, 3)) + SMOOTH)   # topology sensitivity
    return (1 - 2 * tprec * tsens / (tprec + tsens + 1e-8)).mean()


def skeleton_recall_loss(prob: torch.Tensor, fov: torch.Tensor, target_skel: torch.Tensor) -> torch.Tensor:
    """Binary Skeleton Recall: fraction of the (dilated) ground-truth skeleton *missed* by the prediction.

    The skeleton of the ground truth is computed once offline (in the data loader), so only a cheap weighted
    sum over the prediction is needed -- no iterative soft-skeleton in the training loop.
    """
    tube = soft_dilate(target_skel) * fov
    rec = ((prob * fov * tube).sum(dim=(1, 2, 3)) + SMOOTH) / (tube.sum(dim=(1, 2, 3)) + SMOOTH)
    return (1 - rec).mean()


def skeleton_head_loss(skel_logits: torch.Tensor, target_skel: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
    """Auxiliary head: predict the (1-px dilated) skeleton with Dice + BCE."""
    tgt = soft_dilate(target_skel)
    return dice_loss(torch.sigmoid(skel_logits), tgt, fov) + bce_loss(skel_logits, tgt, fov)


# --------------------------------------------------------------------------------------- full objective
class TGLoss:
    """Callable implementing L_total for one training step (all computation in float32)."""

    def __init__(self, cfg):
        l = cfg.loss
        self.lb, self.lt, self.ls, self.lc = float(l.lambda_bce), float(l.lambda_topo), float(l.lambda_skel_head), float(l.lambda_cons)
        self.topo, self.iters = str(l.topo), int(l.skel_iters)
        assert self.topo in ("none", "cldice", "skeleton_recall", "cldice_skeleton_recall"), self.topo

    def supervised(self, out, batch_y, fov, skel) -> dict[str, torch.Tensor]:
        logits, skel_logits = (out if isinstance(out, tuple) else (out, None))
        logits = logits.float()
        prob = torch.sigmoid(logits)
        terms = {"dice": dice_loss(prob, batch_y, fov), "bce": self.lb * bce_loss(logits, batch_y, fov)}
        if "cldice" in self.topo:
            terms["cldice"] = self.lt * soft_cldice_loss(prob, batch_y, fov, skel, self.iters)
        if "skeleton_recall" in self.topo:
            terms["srl"] = self.lt * skeleton_recall_loss(prob, fov, skel)
        if skel_logits is not None and self.topo != "none":
            terms["skel_head"] = self.ls * skeleton_head_loss(skel_logits.float(), skel, fov)
        return terms

    @staticmethod
    def consistency(logits_dg: torch.Tensor, logits_clean: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
        """Mean squared difference of the vessel probabilities (clean view = detached 'teacher')."""
        d = (torch.sigmoid(logits_dg.float()) - torch.sigmoid(logits_clean.float()).detach()) ** 2
        return (d * fov).sum() / fov.sum().clamp(min=1.0)
