import os
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VehiclePoseTrainer:
    """Handles training of YOLOv11 model for vehicle pose estimation"""
    
    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.setup_directories()
        
    def _load_config(self, config_path: str) -> dict:
        """Load training configuration from YAML"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_directories(self):
        """Create necessary directories for training"""
        # Create directories for checkpoints and logs
        self.save_dir = Path(self.config.get('save_dir', 'runs/train'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique run directory
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.save_dir / f'exp_{time_str}'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(self.run_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
    
    def load_model(self):
        """Load YOLOv11 model"""
        try:
            # Load pre-trained YOLOv11 pose model
            self.model = YOLO('yolov11m-pose.pt')
            logger.info("Successfully loaded YOLOv11m-pose model")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def train(self):
        """Train the model"""
        if self.model is None:
            self.load_model()
        
        try:
            # Get training parameters from config
            train_params = {
                'data': self.config['dataset']['yaml_path'],
                'epochs': self.config['training']['epochs'],
                'imgsz': self.config['training']['image_size'],
                'batch': self.config['training']['batch_size'],
                'device': self.config['training'].get('device', 'cuda:0'),
                'workers': self.config['training'].get('num_workers', 8),
                'project': str(self.run_dir),
                'name': 'train',
                'exist_ok': True,
                'pretrained': True,
                'optimizer': self.config['training'].get('optimizer', 'auto'),
                'lr0': self.config['training'].get('learning_rate', 0.01),
                'weight_decay': self.config['training'].get('weight_decay', 0.0005),
                'warmup_epochs': self.config['training'].get('warmup_epochs', 3),
                'close_mosaic': self.config['training'].get('close_mosaic', 10),
                'box': self.config['training'].get('box_loss_weight', 7.5),
                'cls': self.config['training'].get('cls_loss_weight', 0.5),
                'dfl': self.config['training'].get('dfl_loss_weight', 1.5),
                'pose': self.config['training'].get('pose_loss_weight', 12.0),
                'kobj': self.config['training'].get('kpt_obj_weight', 2.0),
            }
            
            # Start training
            logger.info("Starting training with parameters:")
            logger.info(train_params)
            results = self.model.train(**train_params)
            
            # Save training results
            results_path = self.run_dir / 'results.yaml'
            with open(results_path, 'w') as f:
                yaml.dump(results, f)
            
            logger.info(f"Training completed. Results saved to {results_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def validate(self):
        """Validate the model"""
        if self.model is None:
            self.load_model()
        
        try:
            # Get validation parameters
            val_params = {
                'data': self.config['dataset']['yaml_path'],
                'batch': self.config['validation'].get('batch_size', 16),
                'imgsz': self.config['validation'].get('image_size', 640),
                'device': self.config['validation'].get('device', 'cuda:0'),
                'project': str(self.run_dir),
                'name': 'val',
                'exist_ok': True
            }
            
            # Run validation
            logger.info("Starting validation")
            metrics = self.model.val(**val_params)
            
            # Save validation results
            metrics_path = self.run_dir / 'val_metrics.yaml'
            with open(metrics_path, 'w') as f:
                yaml.dump(metrics, f)
            
            logger.info(f"Validation completed. Metrics saved to {metrics_path}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            raise

def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv11 for vehicle pose estimation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = VehiclePoseTrainer(args.config)
    
    # Train and validate
    trainer.train()
    trainer.validate()

if __name__ == '__main__':
    main()
