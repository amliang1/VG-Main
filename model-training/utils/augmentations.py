import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Optional

class Albumentations:
    """
    Custom augmentation pipeline optimized for vehicle pose estimation.
    Focuses on preserving keypoint relationships while applying realistic transforms.
    """
    
    def __init__(self, augment_config: Dict):
        """
        Initialize augmentation pipeline with configurable parameters
        
        Args:
            augment_config: Dictionary containing augmentation parameters
                - degrees: Maximum rotation angle
                - translate: Maximum translation fraction
                - scale: Maximum scale change
                - shear: Maximum shear angle
                - perspective: Maximum perspective distortion
                - flipud: Vertical flip probability
                - fliplr: Horizontal flip probability
                - hsv_h: Hue variation
                - hsv_s: Saturation variation
                - hsv_v: Value variation
        """
        self.transform = A.Compose([
            # Spatial transforms that preserve vehicle geometry
            A.LongestMaxSize(max_size=640, p=1.0),
            A.PadIfNeeded(
                min_height=640,
                min_width=640,
                border_mode=0,
                value=(114, 114, 114),
                p=1.0
            ),
            A.RandomRotate90(p=0.0),  # Disabled as vehicles are usually upright
            A.Rotate(
                limit=augment_config.get('degrees', 10.0),
                interpolation=1,
                border_mode=0,
                p=0.5
            ),
            A.Affine(
                scale=dict(x=(1 - augment_config.get('scale', 0.5),
                           1 + augment_config.get('scale', 0.5))),
                translate_percent=dict(
                    x=(-augment_config.get('translate', 0.2),
                       augment_config.get('translate', 0.2)),
                    y=(-augment_config.get('translate', 0.2),
                       augment_config.get('translate', 0.2))
                ),
                shear=dict(x=(-augment_config.get('shear', 2.0),
                            augment_config.get('shear', 2.0))),
                interpolation=1,
                border_mode=0,
                p=0.5
            ),
            
            # Flips - use with caution as they affect vehicle orientation
            A.HorizontalFlip(p=augment_config.get('fliplr', 0.0)),  # Disabled by default
            A.VerticalFlip(p=augment_config.get('flipud', 0.0)),    # Disabled by default
            
            # Color and lighting transforms
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=augment_config.get('hsv_h', 0.015) * 360,
                    sat_shift_limit=augment_config.get('hsv_s', 0.7) * 100,
                    val_shift_limit=augment_config.get('hsv_v', 0.4) * 100,
                    p=1.0
                ),
            ], p=0.5),
            
            # Realistic vehicle imaging conditions
            A.OneOf([
                # Simulated camera blur
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            
            # Weather and environmental effects
            A.OneOf([
                # Shadows from buildings/trees
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    shadow_dimension=4,
                    p=1.0
                ),
                # Light fog/haze
                A.RandomFog(
                    fog_coef_lower=0.1,
                    fog_coef_upper=0.2,
                    alpha_coef=0.1,
                    p=1.0
                ),
            ], p=0.2),
            
            # Image normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
        ], 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3
        ),
        keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False,
            angle_in_degrees=True
        ))

    def __call__(self, image, bboxes: Optional[list] = None,
                 keypoints: Optional[list] = None,
                 visibility: Optional[list] = None,
                 class_labels: Optional[list] = None) -> Dict:
        """
        Apply augmentation pipeline to image and annotations
        
        Args:
            image: Input image
            bboxes: List of bounding boxes in Pascal VOC format
            keypoints: List of keypoint coordinates
            visibility: List of keypoint visibility flags
            class_labels: List of class labels for bounding boxes
        
        Returns:
            Dictionary containing augmented image and annotations
        """
        # Handle single-class vehicle detection
        if class_labels is None and bboxes is not None:
            class_labels = [0] * len(bboxes)
        
        # Apply transforms
        result = self.transform(
            image=image,
            bboxes=bboxes,
            keypoints=keypoints,
            class_labels=class_labels,
            visibility=visibility
        )
        
        return result
