import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from .mobilenetv3 import MobileNetV3

class ConvBlock(nn.Module):
    """Efficient convolution block with batch normalization"""
    def __init__(self, in_c: int, out_c: int, k: int = 1, s: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, k//2, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """Residual block with channel attention"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels, 3)
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        # Apply channel attention
        b, c, _, _ = out.shape
        y = self.avg_pool(out).view(b, c)
        y = self.ca(y).view(b, c, 1, 1)
        out = out * y.expand_as(out)
        
        return out + identity

class CSPBlock(nn.Module):
    """Cross Stage Partial block"""
    def __init__(self, in_c: int, out_c: int, n: int = 1):
        super().__init__()
        self.conv1 = ConvBlock(in_c, out_c//2)
        self.conv2 = ConvBlock(in_c, out_c//2)
        self.conv3 = ConvBlock(out_c, out_c)
        
        self.blocks = nn.Sequential(*[
            ResidualBlock(out_c//2) for _ in range(n)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        return self.conv3(torch.cat([y1, y2], dim=1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast"""
    def __init__(self, in_c: int, out_c: int, k: int = 5):
        super().__init__()
        c_ = in_c // 2
        self.conv1 = ConvBlock(in_c, c_)
        self.conv2 = ConvBlock(c_ * 4, out_c)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], 1))

class PoseHead(nn.Module):
    """Vehicle pose estimation head"""
    def __init__(self, in_c: int, n_classes: int = 1, n_keypoints: int = 9):
        super().__init__()
        self.n_keypoints = n_keypoints
        
        # Detection head
        self.det_head = nn.Sequential(
            ConvBlock(in_c, in_c),
            ConvBlock(in_c, in_c, 3),
            nn.Conv2d(in_c, n_classes + 4, 1)  # cls + box
        )
        
        # Keypoint head with attention
        self.kpt_feat = nn.Sequential(
            ConvBlock(in_c, in_c),
            ConvBlock(in_c, in_c, 3),
            ConvBlock(in_c, in_c)
        )
        
        # Spatial attention for keypoints
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_c, 1, 1),
            nn.Sigmoid()
        )
        
        # Keypoint prediction
        self.kpt_reg = nn.Conv2d(in_c, n_keypoints * 3, 1)  # x,y,visibility
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Detection branch
        det_out = self.det_head(x)
        
        # Keypoint branch with attention
        kpt_feat = self.kpt_feat(x)
        att_mask = self.spatial_att(kpt_feat)
        kpt_feat = kpt_feat * att_mask
        kpt_out = self.kpt_reg(kpt_feat)
        
        # Reshape keypoint output
        b, _, h, w = kpt_out.shape
        kpt_out = kpt_out.view(b, self.n_keypoints, 3, h, w)
        
        return det_out, kpt_out

class YOLOv11Pose(nn.Module):
    """YOLOv11 with pose estimation for vehicle keypoint detection"""
    def __init__(self, 
                 n_classes: int = 1,
                 n_keypoints: int = 9,
                 width_mult: float = 1.0):
        super().__init__()
        
        # Backbone
        self.backbone = MobileNetV3(width_mult=width_mult)
        c3, c4, c5 = self.backbone.feature_dims
        
        # Neck
        self.sppf = SPPF(c5, c5)
        
        # FPN
        self.lat_c4 = ConvBlock(c4, c4//2)
        self.lat_c3 = ConvBlock(c3, c3//2)
        self.fpn_c4 = CSPBlock(c4, c4)
        self.fpn_c3 = CSPBlock(c3, c3)
        
        # PAN
        self.pan_c4 = CSPBlock(c4, c4)
        self.pan_c5 = CSPBlock(c5, c5)
        
        # Heads for different scales
        self.p3_head = PoseHead(c3, n_classes, n_keypoints)
        self.p4_head = PoseHead(c4, n_classes, n_keypoints)
        self.p5_head = PoseHead(c5, n_classes, n_keypoints)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Backbone
        c3, c4, c5 = self.backbone(x)
        
        # Neck
        p5 = self.sppf(c5)
        
        # FPN
        p4 = self.fpn_c4(torch.cat([
            F.interpolate(p5, size=c4.shape[-2:], mode='nearest'),
            self.lat_c4(c4)
        ], dim=1))
        
        p3 = self.fpn_c3(torch.cat([
            F.interpolate(p4, size=c3.shape[-2:], mode='nearest'),
            self.lat_c3(c3)
        ], dim=1))
        
        # PAN
        p4 = self.pan_c4(torch.cat([
            F.interpolate(p3, size=c4.shape[-2:], mode='nearest'),
            p4
        ], dim=1))
        
        p5 = self.pan_c5(torch.cat([
            F.interpolate(p4, size=c5.shape[-2:], mode='nearest'),
            p5
        ], dim=1))
        
        # Heads
        det3, kpt3 = self.p3_head(p3)
        det4, kpt4 = self.p4_head(p4)
        det5, kpt5 = self.p5_head(p5)
        
        return [det3, det4, det5], [kpt3, kpt4, kpt5]
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Model inference with post-processing"""
        det_out, kpt_out = self(x)
        
        # Process detections
        dets = []
        kpts = []
        for det, kpt in zip(det_out, kpt_out):
            # Get detection outputs
            b, c, h, w = det.shape
            det = det.view(b, -1, c).permute(0, 2, 1)
            dets.append(det)
            
            # Get keypoint outputs
            b, k, _, h, w = kpt.shape
            kpt = kpt.view(b, k, -1, h*w).permute(0, 1, 3, 2)
            kpts.append(kpt)
        
        # Combine predictions from different scales
        dets = torch.cat(dets, dim=2)
        kpts = torch.cat(kpts, dim=2)
        
        return {
            'detections': dets,  # [batch, n_classes + 4, n_anchors]
            'keypoints': kpts    # [batch, n_keypoints, n_anchors, 3]
        }
