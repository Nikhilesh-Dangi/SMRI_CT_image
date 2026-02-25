import numpy as np
import torch


def dice_per_class(pred, target, num_classes, eps=1e-6):
    dices = []
    for c in range(1, num_classes + 1):
        pred_c = (pred == c).float()
        targ_c = (target == c).float()
        inter = (pred_c * targ_c).sum()
        denom = pred_c.sum() + targ_c.sum() + eps
        dices.append((2 * inter + eps) / denom)
    return torch.stack(dices)


def soft_dice_loss(logits, target, num_classes, eps=1e-6):
    # logits: [B,C,D,H,W], target: [B,D,H,W]
    probs = torch.softmax(logits, dim=1)
    loss = 0.0
    for c in range(1, num_classes + 1):
        p = probs[:, c]
        t = (target == c).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum() + eps
        loss += 1 - (2 * inter + eps) / denom
    return loss / num_classes
