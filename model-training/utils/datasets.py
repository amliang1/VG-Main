import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class VehicleDataset(Dataset):
    def __init__(self, data_file: str, transform: Optional[object] = None,
                 img_size: int = 640, is_train: bool = True):
        """
        Dataset class for vehicle detection and pose estimation
        
        Args:
            data_file: Path to text file containing image paths and annotations
            transform: Albumentations transform pipeline
            img_size: Input image size
            is_train: Whether this is training or validation dataset
        """
        self.img_size = img_size
        self.transform = transform
        self.is_train = is_train
        
        # Load dataset
        self.data = self._load_dataset(data_file)
        
        # Set up augmentation parameters
        self.mosaic_border = [-img_size // 2, -img_size // 2]

    def _load_dataset(self, data_file: str) -> List[Dict]:
        """Load dataset from annotation file"""
        dataset = []
        with open(data_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            # Parse line: image_path x1,y1,x2,y2,class_id kp1_x,kp1_y,kp1_v ... kp9_x,kp9_y,kp9_v
            parts = line.strip().split()
            img_path = parts[0]
            bbox = np.array(parts[1].split(','), dtype=np.float32)
            keypoints = np.array([kp.split(',') for kp in parts[2:]], dtype=np.float32)
            
            dataset.append({
                'img_path': img_path,
                'bbox': bbox,  # [x1, y1, x2, y2, class_id]
                'keypoints': keypoints  # [N, 3] - x, y, visibility
            })
        
        return dataset

    def __len__(self) -> int:
        return len(self.data)

    def _load_image(self, index: int) -> np.ndarray:
        """Load and preprocess image"""
        img_path = self.data[index]['img_path']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_mosaic(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load mosaic augmentation"""
        indices = [index] + [np.random.randint(0, len(self.data)) for _ in range(3)]
        yc, xc = [int(np.random.uniform(-x, 2 * self.img_size + x)) for x in self.mosaic_border]
        
        # Initialize mosaic image
        mosaic_img = np.full((self.img_size * 2, self.img_size * 2, 3), 114, dtype=np.uint8)
        
        # Initialize lists for combined annotations
        combined_boxes = []
        combined_keypoints = []
        
        for i, index in enumerate(indices):
            # Load image and annotations
            img = self._load_image(index)
            boxes = self.data[index]['bbox'].copy()
            keypoints = self.data[index]['keypoints'].copy()
            
            # Place image in mosaic
            h, w = img.shape[:2]
            if i == 0:  # top left
                x1a, y1a = max(xc - w, 0), max(yc - h, 0)
                x2a, y2a = xc, yc
            elif i == 1:  # top right
                x1a, y1a = xc, max(yc - h, 0)
                x2a, y2a = min(xc + w, self.img_size * 2), yc
            elif i == 2:  # bottom left
                x1a, y1a = max(xc - w, 0), yc
                x2a, y2a = xc, min(yc + h, self.img_size * 2)
            else:  # bottom right
                x1a, y1a = xc, yc
                x2a, y2a = min(xc + w, self.img_size * 2), min(yc + h, self.img_size * 2)
            
            # Calculate source image coordinates
            x1b = w - (x2a - x1a)
            y1b = h - (y2a - y1a)
            x2b = w
            y2b = h
            
            # Place image
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # Adjust coordinates
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * w + x1a
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * h + y1a
            keypoints[:, 0] = keypoints[:, 0] * w + x1a
            keypoints[:, 1] = keypoints[:, 1] * h + y1a
            
            combined_boxes.append(boxes)
            combined_keypoints.append(keypoints)
        
        # Combine annotations
        boxes = np.concatenate(combined_boxes, 0)
        keypoints = np.concatenate(combined_keypoints, 0)
        
        # Clip to mosaic boundaries
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, 2 * self.img_size)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, 2 * self.img_size)
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, 2 * self.img_size)
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, 2 * self.img_size)
        
        return mosaic_img, boxes, keypoints

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get item from dataset"""
        # Load image and annotations
        if self.is_train and np.random.random() < 0.5:
            img, boxes, keypoints = self._load_mosaic(index)
        else:
            img = self._load_image(index)
            boxes = self.data[index]['bbox'].copy()
            keypoints = self.data[index]['keypoints'].copy()
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(
                image=img,
                bboxes=boxes,
                keypoints=keypoints[:, :2],
                visibility=keypoints[:, 2]
            )
            img = transformed['image']
            boxes = np.array(transformed['bboxes'])
            keypoints = np.array(transformed['keypoints'])
            visibility = np.array(transformed['visibility'])
            
            # Combine keypoints and visibility
            keypoints = np.concatenate([keypoints, visibility[:, None]], axis=1)
        
        # Convert to tensors
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        
        # Prepare target tensors
        target = {
            'boxes': torch.from_numpy(boxes).float(),
            'keypoints': torch.from_numpy(keypoints).float(),
            'labels': torch.ones(len(boxes), dtype=torch.long),  # Only vehicle class
            'visibility': torch.from_numpy(visibility).float()
        }
        
        return img, target
