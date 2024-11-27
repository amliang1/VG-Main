import torch
import numpy as np
from typing import Dict, List, Tuple

def compute_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Compute IoU between predicted and target boxes"""
    # Extract coordinates
    pred_left = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    pred_right = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    pred_top = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    pred_bottom = pred_boxes[..., 1] + pred_boxes[..., 3] / 2
    
    target_left = target_boxes[..., 0] - target_boxes[..., 2] / 2
    target_right = target_boxes[..., 0] + target_boxes[..., 2] / 2
    target_top = target_boxes[..., 1] - target_boxes[..., 3] / 2
    target_bottom = target_boxes[..., 1] + target_boxes[..., 3] / 2
    
    # Compute intersection
    intersect_left = torch.max(pred_left, target_left)
    intersect_right = torch.min(pred_right, target_right)
    intersect_top = torch.max(pred_top, target_top)
    intersect_bottom = torch.min(pred_bottom, target_bottom)
    
    intersect_width = (intersect_right - intersect_left).clamp(min=0)
    intersect_height = (intersect_bottom - intersect_top).clamp(min=0)
    intersect_area = intersect_width * intersect_height
    
    # Compute union
    pred_area = (pred_right - pred_left) * (pred_bottom - pred_top)
    target_area = (target_right - target_left) * (target_bottom - target_top)
    union_area = pred_area + target_area - intersect_area
    
    # Compute IoU
    iou = intersect_area / (union_area + 1e-7)
    
    return iou

def compute_oks(pred_kpts: torch.Tensor, target_kpts: torch.Tensor,
                visibility: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """Compute OKS between predicted and target keypoints"""
    # Reshape keypoints
    pred_kpts = pred_kpts.view(-1, pred_kpts.size(-1) // 3, 3)
    target_kpts = target_kpts.view(-1, target_kpts.size(-1) // 3, 3)
    visibility = visibility.view(-1, visibility.size(-1))
    
    # Compute squared distances
    dx = pred_kpts[..., 0] - target_kpts[..., 0]
    dy = pred_kpts[..., 1] - target_kpts[..., 1]
    dv = pred_kpts[..., 2].sigmoid() - target_kpts[..., 2]
    
    d = (dx ** 2 + dy ** 2) / (2 * sigmas ** 2)
    
    # Apply visibility mask
    valid_mask = visibility > 0
    d = d * valid_mask.float()
    
    # Compute OKS
    oks = torch.exp(-d)
    oks = oks.mean(dim=1)  # Average over keypoints
    
    return oks

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using 11-point interpolation"""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap

def compute_metrics(predictions: Dict[str, torch.Tensor],
                   targets: Dict[str, torch.Tensor],
                   iou_threshold: float = 0.5,
                   oks_threshold: float = 0.5) -> Dict[str, float]:
    """Compute detection and pose estimation metrics"""
    device = predictions['cls'].device
    
    # Move targets to device
    targets = {k: v.to(device) for k, v in targets.items()}
    
    # Get objectness and class predictions
    obj_pred = torch.sigmoid(predictions['obj'])
    cls_pred = torch.sigmoid(predictions['cls'])
    
    # Compute detection metrics
    iou = compute_iou(predictions['reg'], targets['reg'])
    detection_mask = (iou > iou_threshold) & (obj_pred > 0.5)
    
    # Compute pose metrics
    oks = compute_oks(
        predictions['pose'],
        targets['pose'],
        targets['visibility'],
        torch.ones(9, device=device) * 0.05  # Fixed sigmas for now
    )
    pose_mask = oks > oks_threshold
    
    # Compute precision and recall
    true_positives = detection_mask & pose_mask
    false_positives = ~detection_mask & (obj_pred > 0.5)
    false_negatives = ~true_positives & (targets['obj'] > 0.5)
    
    precision = true_positives.sum().float() / (true_positives.sum() + false_positives.sum() + 1e-7)
    recall = true_positives.sum().float() / (true_positives.sum() + false_negatives.sum() + 1e-7)
    
    # Compute F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    # Compute mean IoU and OKS
    mean_iou = iou[detection_mask].mean() if detection_mask.any() else torch.tensor(0.0)
    mean_oks = oks[pose_mask].mean() if pose_mask.any() else torch.tensor(0.0)
    
    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'map': compute_ap(
            recall.cpu().numpy(),
            precision.cpu().numpy()
        ),
        'mean_iou': mean_iou.item(),
        'mean_oks': mean_oks.item()
    }
