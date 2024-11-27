import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class DistillationLoss(nn.Module):
    """Knowledge Distillation Loss combining hard and soft targets"""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute distillation loss
        
        Args:
            student_logits: Output from student model
            teacher_logits: Output from teacher model
            targets: Ground truth labels
        """
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean')
        
        # Hard targets loss
        hard_loss = F.cross_entropy(student_logits, targets)
        
        # Combine losses
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss * (self.temperature ** 2)

class TeacherStudentTrainer:
    """Handles knowledge distillation training process"""
    
    def __init__(self, teacher_model: nn.Module,
                 student_model: nn.Module,
                 config: Dict):
        self.teacher = teacher_model
        self.student = student_model
        self.config = config
        self.distill_loss = DistillationLoss(
            temperature=config.get('temperature', 4.0),
            alpha=config.get('alpha', 0.5)
        )
        
    def train_step(self, images: torch.Tensor,
                   targets: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Single training step with knowledge distillation"""
        # Teacher predictions (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher(images)
        
        # Student predictions
        student_logits = self.student(images)
        
        # Compute loss
        loss = self.distill_loss(student_logits, teacher_logits, targets)
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {'loss': loss.item()}
    
    def create_compact_student(self) -> nn.Module:
        """Create a compact student model based on teacher architecture"""
        # Example: Reduce number of channels/layers
        compact_config = {
            'depth_multiple': 0.33,  # Reduce depth
            'width_multiple': 0.5,   # Reduce width
            'num_classes': self.config['num_classes']
        }
        
        return self._build_compact_model(compact_config)
    
    def _build_compact_model(self, config: Dict) -> nn.Module:
        """Build compact student model (implementation depends on base architecture)"""
        # This should be implemented based on your specific architecture
        raise NotImplementedError
    
    def evaluate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate student model performance"""
        self.student.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                # Get predictions
                student_logits = self.student(images)
                teacher_logits = self.teacher(images)
                
                # Compute loss
                loss = self.distill_loss(student_logits, teacher_logits, targets)
                total_loss += loss.item()
                
                # Compute accuracy
                _, predicted = student_logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': 100. * correct / total
        }
