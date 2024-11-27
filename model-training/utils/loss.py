import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

def focal_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0,
               alpha: float = 0.25) -> torch.Tensor:
    """Compute focal loss for classification"""
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p_t = target * torch.sigmoid(pred) + (1 - target) * (1 - torch.sigmoid(pred))
    loss = ce_loss * ((1 - p_t) ** gamma)
    
    if alpha >= 0:
        alpha_t = target * alpha + (1 - target) * (1 - alpha)
        loss = alpha_t * loss
    
    return loss.mean()

def iou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute IoU loss for bounding box regression"""
    # Extract coordinates
    pred_left = pred[..., 0] - pred[..., 2] / 2
    pred_right = pred[..., 0] + pred[..., 2] / 2
    pred_top = pred[..., 1] - pred[..., 3] / 2
    pred_bottom = pred[..., 1] + pred[..., 3] / 2
    
    target_left = target[..., 0] - target[..., 2] / 2
    target_right = target[..., 0] + target[..., 2] / 2
    target_top = target[..., 1] - target[..., 3] / 2
    target_bottom = target[..., 1] + target[..., 3] / 2
    
    # Compute intersection area
    intersect_left = torch.max(pred_left, target_left)
    intersect_right = torch.min(pred_right, target_right)
    intersect_top = torch.max(pred_top, target_top)
    intersect_bottom = torch.min(pred_bottom, target_bottom)
    
    intersect_width = (intersect_right - intersect_left).clamp(min=0)
    intersect_height = (intersect_bottom - intersect_top).clamp(min=0)
    intersect_area = intersect_width * intersect_height
    
    # Compute union area
    pred_area = (pred_right - pred_left) * (pred_bottom - pred_top)
    target_area = (target_right - target_left) * (target_bottom - target_top)
    union_area = pred_area + target_area - intersect_area
    
    # Compute IoU
    iou = intersect_area / (union_area + eps)
    loss = 1 - iou
    
    return loss.mean()

def oks_loss(pred: torch.Tensor, target: torch.Tensor, visibility: torch.Tensor,
             sigmas: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute OKS (Object Keypoint Similarity) loss for pose estimation"""
    # Reshape predictions and targets
    pred = pred.view(-1, pred.size(-1) // 3, 3)  # [B*N, K, 3]
    target = target.view(-1, target.size(-1) // 3, 3)  # [B*N, K, 3]
    visibility = visibility.view(-1, visibility.size(-1))  # [B*N, K]
    
    # Extract coordinates and compute squared distances
    dx = pred[..., 0] - target[..., 0]
    dy = pred[..., 1] - target[..., 1]
    dv = pred[..., 2].sigmoid() - target[..., 2]
    
    d = (dx ** 2 + dy ** 2) / (2 * sigmas ** 2)
    
    # Apply visibility mask
    valid_mask = visibility > 0
    d = d * valid_mask.float()
    
    # Compute OKS
    oks = torch.exp(-d)
    oks = oks.mean(dim=1)  # Average over keypoints
    loss = 1 - oks
    
    return loss.mean()

class PoseLoss(nn.Module):
    def __init__(self, num_classes: int = 1, num_keypoints: int = 9,
                 cls_weight: float = 1.0, obj_weight: float = 1.0,
                 box_weight: float = 5.0, pose_weight: float = 10.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        self.box_weight = box_weight
        self.pose_weight = pose_weight
        
        # Initialize keypoint sigmas (can be learned or set manually)
        self.register_buffer('sigmas', torch.ones(num_keypoints) * 0.05)

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute combined loss for vehicle pose estimation"""
        device = predictions['cls'].device
        
        # Classification loss
        cls_loss = focal_loss(
            predictions['cls'],
            targets['cls'].to(device)
        )
        
        # Objectness loss
        obj_loss = focal_loss(
            predictions['obj'],
            targets['obj'].to(device),
            gamma=1.0
        )
        
        # Box regression loss
        box_loss = iou_loss(
            predictions['reg'],
            targets['reg'].to(device)
        )
        
        # Pose estimation loss
        pose_loss = oks_loss(
            predictions['pose'],
            targets['pose'].to(device),
            targets['visibility'].to(device),
            self.sigmas
        )
        
        # Combine losses
        total_loss = (
            self.cls_weight * cls_loss +
            self.obj_weight * obj_loss +
            self.box_weight * box_loss +
            self.pose_weight * pose_loss
        )
        
        return {
            'loss': total_loss,
            'cls_loss': cls_loss,
            'obj_loss': obj_loss,
            'box_loss': box_loss,
            'pose_loss': pose_loss
        }
