import torch
import torch.nn as nn
import torch.quantization as quantization
from typing import Dict, Optional

class QuantizedModel:
    """Handles model quantization for efficient deployment on Jetson Nano"""
    
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.backend = 'qnnpack'  # Optimized for ARM architectures
        
    def prepare_for_quantization(self):
        """Prepare model for quantization by fusing operations"""
        self.model.eval()
        
        # Fuse Conv, BN, Relu layers
        self.model = torch.quantization.fuse_modules(
            self.model,
            [['conv', 'bn', 'act']],
            inplace=True
        )
        
        # Set qconfig
        self.model.qconfig = torch.quantization.get_default_qconfig(self.backend)
        torch.backends.quantized.engine = self.backend
        
        return torch.quantization.prepare(self.model)
    
    def quantize(self, calibration_loader: Optional[torch.utils.data.DataLoader] = None) -> nn.Module:
        """Quantize the model using calibration data if provided"""
        if calibration_loader is not None:
            # Run calibration
            with torch.no_grad():
                for images, _ in calibration_loader:
                    self.model(images)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(self.model)
        
        # Optimize for inference
        quantized_model.eval()
        
        return quantized_model
    
    def export_torchscript(self, model: nn.Module, save_path: str):
        """Export quantized model to TorchScript format"""
        # Create example input
        example_input = torch.randn(1, 3, 640, 640)
        
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
        
        # Save the traced model
        torch.jit.save(traced_model, save_path)
        
    def benchmark(self, model: nn.Module, input_size: tuple = (1, 3, 640, 640),
                 num_runs: int = 100) -> Dict[str, float]:
        """Benchmark model inference speed"""
        device = next(model.parameters()).device
        input_tensor = torch.randn(input_size).to(device)
        
        # Warmup
        for _ in range(10):
            model(input_tensor)
        
        # Benchmark
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                starter.record()
                model(input_tensor)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
        
        avg_time = sum(timings) / len(timings)
        fps = 1000 / avg_time  # Convert ms to fps
        
        return {
            'avg_inference_time_ms': avg_time,
            'fps': fps,
            'min_time_ms': min(timings),
            'max_time_ms': max(timings)
        }
