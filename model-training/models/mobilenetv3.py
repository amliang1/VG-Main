import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional

class HardSigmoid(nn.Module):
    """Implements the hard sigmoid activation function"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3) / 6

class HardSwish(nn.Module):
    """Implements the hard swish activation function"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3) / 6

class SEModule(nn.Module):
    """Squeeze-and-Excitation attention module"""
    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            HardSigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBNActivation(nn.Module):
    """Standard convolution with BN and activation"""
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 activation: Optional[str] = 'HardSwish'):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if activation == 'HardSwish':
            self.activation = HardSwish()
        elif activation == 'ReLU':
            self.activation = nn.ReLU6(inplace=True)
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class MobileBottleneckV3(nn.Module):
    """MobileNetV3 Bottleneck"""
    def __init__(self,
                 in_channels: int,
                 exp_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 use_se: bool = True,
                 activation: str = 'HardSwish'):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        # Expansion
        if exp_channels != in_channels:
            layers.append(ConvBNActivation(
                in_channels,
                exp_channels,
                1,
                activation=activation
            ))
        
        # Depthwise
        layers.extend([
            # Depthwise conv
            ConvBNActivation(
                exp_channels,
                exp_channels,
                kernel_size,
                stride=stride,
                groups=exp_channels,
                activation=activation
            ),
            # SE module
            SEModule(exp_channels) if use_se else nn.Identity(),
            # Projection
            ConvBNActivation(
                exp_channels,
                out_channels,
                1,
                activation=None
            )
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)

class MobileNetV3(nn.Module):
    """MobileNetV3 backbone for YOLO"""
    def __init__(self,
                 width_mult: float = 1.0,
                 depth_mult: float = 1.0,
                 num_classes: int = 1000):
        super().__init__()
        
        # Configuration for MobileNetV3-Small
        self.cfgs = [
            # k, exp, c,  se,     nl,         s
            [3, 16,  16,  True,   'ReLU',     2],
            [3, 72,  24,  False,  'ReLU',     2],
            [3, 88,  24,  False,  'ReLU',     1],
            [5, 96,  40,  True,   'HardSwish', 2],
            [5, 240, 40,  True,   'HardSwish', 1],
            [5, 240, 40,  True,   'HardSwish', 1],
            [5, 120, 48,  True,   'HardSwish', 1],
            [5, 144, 48,  True,   'HardSwish', 1],
            [5, 288, 96,  True,   'HardSwish', 2],
            [5, 576, 96,  True,   'HardSwish', 1],
            [5, 576, 96,  True,   'HardSwish', 1],
        ]

        input_channels = 16
        features: List[nn.Module] = [
            ConvBNActivation(3, input_channels, 3, stride=2, activation='HardSwish')
        ]

        # Build blocks
        for k, exp, c, se, nl, s in self.cfgs:
            output_channels = self._make_divisible(c * width_mult)
            exp_channels = self._make_divisible(exp * width_mult)
            features.append(MobileBottleneckV3(
                input_channels,
                exp_channels,
                output_channels,
                k,
                s,
                use_se=se,
                activation=nl
            ))
            input_channels = output_channels

        self.features = nn.Sequential(*features)
        
        # Feature dimension
        self.feature_dims = [24, 40, 96]  # C3, C4, C5 feature dimensions
        
    def _make_divisible(self, v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
        """Ensure channel counts are divisible by 8 for hardware efficiency"""
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass returning multiple feature maps for YOLO"""
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Save feature maps at specific stages for YOLO
            if i in [3, 6, 10]:  # Indices for C3, C4, C5
                features.append(x)
        return tuple(features)

class YOLOv11MobileNet(nn.Module):
    """YOLOv11 with MobileNetV3 backbone"""
    def __init__(self, num_classes: int = 80, width_mult: float = 1.0):
        super().__init__()
        self.backbone = MobileNetV3(width_mult=width_mult)
        
        # Neck (FPN + PAN)
        self.neck = self._build_neck()
        
        # Detection head
        self.head = self._build_head(num_classes)
    
    def _build_neck(self) -> nn.Module:
        """Build FPN + PAN neck"""
        # Implement neck architecture
        # This should be customized based on your specific needs
        pass
    
    def _build_head(self, num_classes: int) -> nn.Module:
        """Build detection head"""
        # Implement detection head
        # This should be customized based on your specific needs
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)
        
        # Neck
        neck_features = self.neck(features)
        
        # Head
        return self.head(neck_features)
