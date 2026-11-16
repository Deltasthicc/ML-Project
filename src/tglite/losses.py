"""Training objective   L = L_Dice + l1 L_BCE + l2 L_topo + l3 L_cons    (all terms inside the FOV).

L_topo = topology loss (soft-clDice | Skeleton Recall | cbDice) + l_skel * skeleton-head loss, optionally ramped up
linearly over ``loss.topo_warmup_iters``. All tensors are (B, 1, H, W). Re-implemented from the papers' equations.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

SMOOTH = 1.0


def soft_erode(x):
    return torch.minimum(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def soft_skeleton(x, iters: int):
    """Differentiable morphological skeleton (Shit et al., CVPR 2021, Alg. 1)."""
    skel = F.relu(x - soft_dilate(soft_erode(x)))
    for _ in range(iters):
        x = soft_erode(x)
        delta = F.relu(x - soft_dilate(soft_erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def _sum(t):
    return t.sum(dim=(1, 2, 3))


def dice_loss(prob, target, fov):
    p, t = prob * fov, target * fov
    return (1 - (2 * _sum(p * t) + SMOOTH) / (_sum(p) + _sum(t) + SMOOTH)).mean()


def bce_loss(logits, target, fov):
    l = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (l * fov).sum() / fov.sum().clamp(min=1.0)


def soft_cldice_loss(prob, target, fov, target_skel, iters: int = 10):
    """1 - clDice with the soft skeleton of the prediction and the exact skeleton of the label."""
    p, t = prob * fov, target * fov
    sp, st = soft_skeleton(p, iters) * fov, target_skel * fov
    tprec = (_sum(sp * t) + SMOOTH) / (_sum(sp) + SMOOTH)
    tsens = (_sum(st * p) + SMOOTH) / (_sum(st) + SMOOTH)
    return (1 - 2 * tprec * tsens / (tprec + tsens + 1e-8)).mean()


def skeleton_recall_loss(prob, fov, target_skel):
    """Binary Skeleton Recall (Kirchhoff et al., ECCV 2024): share of the (dilated) label skeleton missed by the prediction."""
    tube = soft_dilate(target_skel) * fov
    return (1 - (_sum(prob * fov * tube) + SMOOTH) / (_sum(tube) + SMOOTH)).mean()


def soft_depth(x, iters: int):
    """Differentiable distance-to-boundary: 1 + number of (alternating cross/square) soft erosions a pixel survives."""
    depth, cur = x, x
    for k in range(iters):
        cur = soft_erode(cur) if k % 2 == 0 else -F.max_pool2d(-cur, 3, 1, 1)
        depth = depth + cur
    return depth


def cbdice_loss(prob, target, fov, target_skel, target_dist, iters: int = 10):
    """1 - cbDice (Shi et al., MICCAI 2024, 2-D cl-MIN form): radius- and boundary-aware clDice."""
    p = prob * fov
    sp, dp = soft_skeleton(p, iters) * fov, soft_depth(p, iters) * fov
    sl, dl = target_skel * fov, target_dist * fov
    rl = dl * sl
    rmax = rl.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
    rp, rl_s = dp.detach().clamp(min=1.0), rl.clamp(min=1.0)
    dlc, dpc = torch.minimum(dl, rmax), torch.minimum(dp, rmax)
    tprec = (_sum(sp * dlc / rp) + SMOOTH) / (_sum(sp * ((1 - sl) + sl * rl_s / rp)) + SMOOTH)
    tsens = (_sum(sl * dpc / rl_s) + SMOOTH) / (_sum(sl * (1 - sp)) + _sum(sl * sp * rp / rl_s) + SMOOTH)
    return (1 - 2 * tprec * tsens / (tprec + tsens + 1e-8)).mean()


def skeleton_head_loss(skel_logits, target_skel, fov):
    tgt = soft_dilate(target_skel)
    return dice_loss(torch.sigmoid(skel_logits), tgt, fov) + bce_loss(skel_logits, tgt, fov)


def consistency_loss(logits_dg, logits_clean, fov):
    """MSE between perturbed-view and (detached) clean-view probabilities."""
    d = (torch.sigmoid(logits_dg.float()) - torch.sigmoid(logits_clean.float()).detach()) ** 2
    return (d * fov).sum() / fov.sum().clamp(min=1.0)


def soft_cldice_between(prob_a, prob_b, fov, iters: int = 10):
    """1 - clDice between two SOFT maps, using the soft skeletons of both (no binary label involved)."""
    a, b = prob_a * fov, prob_b * fov
    sa, sb = soft_skeleton(a, iters) * fov, soft_skeleton(b, iters) * fov
    tprec = (_sum(sa * b) + SMOOTH) / (_sum(sa) + SMOOTH)
    tsens = (_sum(sb * a) + SMOOTH) / (_sum(sb) + SMOOTH)
    return (1 - 2 * tprec * tsens / (tprec + tsens + 1e-8)).mean()


def kd_pixel_loss(logits, teacher, fov):
    """Pixel distillation: BCE between the student logits and the teacher's soft probabilities (inside the FOV)."""
    return bce_loss(logits, teacher, fov)


def kd_skeleton_loss(prob, teacher, fov, iters: int = 10):
    """v2.1 novelty 3 - topology-aware distillation: the student's soft centreline must lie on the teacher's vessels and
    the teacher's soft centreline must be covered by the student (soft-clDice between student and teacher)."""
    return soft_cldice_between(prob, teacher, fov, iters)


def skeleton_consistency_loss(logits_dg, logits_clean, fov, iters: int = 10):
    """v2.1 novelty 4 - the perturbed view must reproduce the CENTRELINES of the (detached) clean view, not every pixel."""
    return soft_cldice_between(torch.sigmoid(logits_dg.float()), torch.sigmoid(logits_clean.float()).detach(), fov, iters)


def topo_scale(it: int, warmup: int) -> float:
    return 1.0 if warmup <= 0 else min(1.0, (it + 1) / warmup)


class Objective:
    def __init__(self, cfg):
        l = cfg.loss
        self.lb, self.lt, self.ls, self.lc = float(l.lambda_bce), float(l.lambda_topo), float(l.lambda_skel_head), float(l.lambda_cons)
        self.topo, self.iters, self.warmup = str(l.topo), int(l.skel_iters), int(l.get("topo_warmup_iters", 0) or 0)
        assert self.topo in ("none", "cldice", "skeleton_recall", "cbdice"), self.topo
        self.cons_mode = str(l.get("cons_mode", "pixel"))
        assert self.cons_mode in ("pixel", "skeleton"), self.cons_mode
        self.l_aux = float(l.get("lambda_aux", 0.5))
        kd = cfg.get("distill") or {}
        self.kd = bool(kd.get("enabled", False))
        self.l_kd, self.l_kds = float(kd.get("lambda_kd", 0.0)), float(kd.get("lambda_kd_skel", 0.0))

    def consistency(self, logits_dg, logits_clean, fov):
        if self.cons_mode == "skeleton":
            return self.lc * skeleton_consistency_loss(logits_dg, logits_clean, fov, self.iters)
        return self.lc * consistency_loss(logits_dg, logits_clean, fov)

    def supervised(self, out, y, fov, skel, dist=None, it: int = 0, teacher=None, aux=None) -> dict:
        logits, skel_logits = out if isinstance(out, tuple) else (out, None)
        logits = logits.float()
        prob = torch.sigmoid(logits)
        terms = {"dice": dice_loss(prob, y, fov), "bce": self.lb * bce_loss(logits, y, fov)}
        ts = topo_scale(it, self.warmup)
        if self.topo == "cldice":
            terms["topo"] = ts * self.lt * soft_cldice_loss(prob, y, fov, skel, self.iters)
        elif self.topo == "skeleton_recall":
            terms["topo"] = ts * self.lt * skeleton_recall_loss(prob, fov, skel)
        elif self.topo == "cbdice":
            terms["topo"] = ts * self.lt * cbdice_loss(prob, y, fov, skel, dist, self.iters)
        if skel_logits is not None and self.topo != "none" and self.ls > 0:
            terms["skel_head"] = ts * self.ls * skeleton_head_loss(skel_logits.float(), skel, fov)
        if aux is not None and self.l_aux > 0:              # W-Net first stage
            a = aux.float()
            terms["aux"] = self.l_aux * (dice_loss(torch.sigmoid(a), y, fov) + self.lb * bce_loss(a, y, fov))
        if self.kd and teacher is not None:
            if self.l_kd > 0:
                terms["kd"] = self.l_kd * kd_pixel_loss(logits, teacher, fov)
            if self.l_kds > 0:
                terms["kd_skel"] = self.l_kds * kd_skeleton_loss(prob, teacher, fov, self.iters)
        return terms
