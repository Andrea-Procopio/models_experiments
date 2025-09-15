import torch
import torch.nn.functional as F

def dice_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,2,h,w); labels: (B,h,w) in {0,1}
    Dice on foreground channel with eps smoothing.
    """
    # Ensure contiguous and device-aligned for stable backward on MPS/CPU
    if labels.device != logits.device:
        labels = labels.to(logits.device)
    logits = logits.contiguous()
    labels = labels.contiguous()
    probs = torch.softmax(logits, dim=1)[:, 1].contiguous()  # (B,h,w)
    y = (labels == 1).float().contiguous()
    eps = 1e-6
    inter = (probs * y).sum(dim=(-2, -1))
    denom = probs.sum(dim=(-2, -1)) + y.sum(dim=(-2, -1))
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()
