import numpy as np
from scipy import ndimage
from skimage.morphology import closing, dilation, disk, erosion
from skimage.measure import label

def confusion_binary(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    return tp, fp, fn, tn

def iou_dice(pred: np.ndarray, gt: np.ndarray):
    tp, fp, fn, _ = confusion_binary(pred, gt)
    denom_iou = tp + fp + fn
    iou = (tp / denom_iou) if denom_iou else 0.0
    denom_d = 2*tp + fp + fn
    dice = (2*tp / denom_d) if denom_d else 0.0
    return iou, dice

def boundary_f1(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool); gt = gt.astype(bool)
    # edges via 1-px erosion xor (simple, consistent)
    pred_edge = pred ^ ndimage.binary_erosion(pred)
    gt_edge   = gt ^ ndimage.binary_erosion(gt)
    dt_pred = ndimage.distance_transform_edt(~pred_edge)
    dt_gt   = ndimage.distance_transform_edt(~gt_edge)
    tol = max(1, int(0.003 * max(pred.shape)))  # ~0.3% of max dim
    # precision
    tp_p = (dt_gt[pred_edge] <= tol).sum()
    prec = tp_p / max(1, pred_edge.sum())
    # recall
    tp_r = (dt_pred[gt_edge] <= tol).sum()
    rec  = tp_r / max(1, gt_edge.sum())
    return (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0

def concavity_fill_index(pred: np.ndarray, gt: np.ndarray, r: int):
    """
    CFI_r = |pred ∧ (close(gt,r) \ gt)| / |close(gt,r) \ gt|
    """
    if r <= 0:
        return 0.0
    C = closing(gt.astype(bool), footprint=disk(r))
    gap = (C & ~gt.astype(bool))
    denom = int(gap.sum())
    if denom == 0:
        return 0.0
    num = int((pred.astype(bool) & gap).sum())
    return num / denom

def overfill_index(pred: np.ndarray, gt: np.ndarray, r: int):
    """OFI_r: outward bleed into the outer band."""
    if r <= 0: return 0.0
    band = np.logical_and(dilation(gt.astype(bool), footprint=disk(r)),
                          ~gt.astype(bool))
    denom = int(band.sum())
    if denom == 0: return 0.0
    num = int(np.logical_and(pred.astype(bool), band).sum())
    return num / denom

def underfill_index(pred: np.ndarray, gt: np.ndarray, r: int):
    """UFI_r: inward erosion inside the inner band."""
    if r <= 0: return 0.0
    band = np.logical_and(gt.astype(bool),
                          ~erosion(gt.astype(bool), footprint=disk(r)))
    denom = int(band.sum())
    if denom == 0: return 0.0
    num = int(np.logical_and(~pred.astype(bool), band).sum())
    return num / denom