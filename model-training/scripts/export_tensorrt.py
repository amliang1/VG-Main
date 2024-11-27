#!/usr/bin/env python3
import torch
import logging
from pathlib import Path
import json
from ultralytics import YOLO
import tensorrt as trt
import numpy as np
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorRTExporter:
    """Export and optimize YOLO model for TensorRT deployment"""
    
    def __init__(self, model_path, config_path):
        """Initialize exporter with model and configuration"""
        self.model = YOLO(model_path)
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path):
        """Load export configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def export(self, output_path, precision='fp16'):
        """Export model to TensorRT format"""
        logger.info(f"Exporting model to TensorRT with {precision} precision...")
        
        try:
            # Export to ONNX first
            onnx_path = str(Path(output_path).with_suffix('.onnx'))
            self.model.export(format='onnx', 
                            dynamic=True,
                            simplify=True,
                            opset=12,
                            path=onnx_path)
            
            # Convert ONNX to TensorRT
            trt_path = str(Path(output_path).with_suffix('.engine'))
            self.model.export(format='engine',
                            device=0,
                            half=precision=='fp16',
                            workspace=8,
                            verbose=False,
                            path=trt_path)
            
            logger.info(f"Successfully exported model to {trt_path}")
            return trt_path
            
        except Exception as e:
            logger.error(f"Error during export: {str(e)}")
            raise
    
    def optimize(self, engine_path):
        """Apply TensorRT optimizations"""
        logger.info("Applying TensorRT optimizations...")
        
        try:
            # Load TensorRT engine
            with open(engine_path, 'rb') as f:
                engine_str = f.read()
            
            # Create TensorRT runtime and engine
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine = runtime.deserialize_cuda_engine(engine_str)
            
            # Apply optimizations
            self._optimize_workspace(engine)
            self._optimize_dla(engine)
            self._optimize_precision(engine)
            
            logger.info("Successfully applied TensorRT optimizations")
            
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise
    
    def _optimize_workspace(self, engine):
        """Optimize TensorRT workspace memory"""
        workspace_size = self.config.get('workspace_size', 8) # GB
        engine.max_workspace_size = workspace_size * (1 << 30)
    
    def _optimize_dla(self, engine):
        """Optimize for Deep Learning Accelerator if available"""
        if self.config.get('use_dla', False):
            engine.DLA_core = 0
            engine.default_device_type = trt.DeviceType.DLA
    
    def _optimize_precision(self, engine):
        """Optimize precision settings"""
        if self.config.get('precision', 'fp16') == 'fp16':
            engine.strict_type_constraints = True
            engine.clear_flag(trt.BuilderFlag.TF32)
    
    def benchmark(self, engine_path, input_shape=(1, 3, 640, 640), num_iterations=100):
        """Benchmark the exported TensorRT engine"""
        logger.info("Running benchmark...")
        
        try:
            # Load engine
            with open(engine_path, 'rb') as f:
                engine_str = f.read()
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine = runtime.deserialize_cuda_engine(engine_str)
            
            # Create execution context
            context = engine.create_execution_context()
            
            # Prepare input
            input_data = np.random.rand(*input_shape).astype(np.float32)
            input_tensor = torch.from_numpy(input_data).cuda()
            
            # Warmup
            for _ in range(10):
                context.execute_v2([input_tensor.data_ptr()])
            
            # Benchmark
            times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                context.execute_v2([input_tensor.data_ptr()])
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            
            # Calculate statistics
            mean_time = np.mean(times)
            std_time = np.std(times)
            fps = 1.0 / mean_time
            
            results = {
                'mean_inference_time': mean_time,
                'std_inference_time': std_time,
                'fps': fps,
                'input_shape': input_shape,
                'precision': self.config.get('precision', 'fp16')
            }
            
            logger.info(f"Benchmark results: {json.dumps(results, indent=2)}")
            return results
            
        except Exception as e:
            logger.error(f"Error during benchmarking: {str(e)}")
            raise
    
    def validate_engine(self, engine_path, val_data):
        """Validate the exported TensorRT engine"""
        logger.info("Validating TensorRT engine...")
        
        try:
            # Load validation data
            val_loader = self._prepare_validation_data(val_data)
            
            # Load engine
            with open(engine_path, 'rb') as f:
                engine_str = f.read()
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine = runtime.deserialize_cuda_engine(engine_str)
            context = engine.create_execution_context()
            
            # Run validation
            metrics = self._run_validation(context, val_loader)
            
            logger.info(f"Validation results: {json.dumps(metrics, indent=2)}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            raise
    
    def _prepare_validation_data(self, val_data):
        """Prepare validation data loader"""
        # Implementation for validation data preparation
        pass
    
    def _run_validation(self, context, val_loader):
        """Run validation on TensorRT engine"""
        # Implementation for validation
        pass

def main():
    """Main export script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export YOLO model to TensorRT')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--config', type=str, required=True, help='Path to export config')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16', 'int8'],
                        help='TensorRT precision')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark after export')
    parser.add_argument('--validate', action='store_true', help='Validate exported engine')
    parser.add_argument('--val-data', type=str, help='Path to validation data')
    args = parser.parse_args()
    
    # Initialize exporter
    exporter = TensorRTExporter(args.model, args.config)
    
    # Export model
    engine_path = exporter.export(args.output, args.precision)
    
    # Optimize engine
    exporter.optimize(engine_path)
    
    # Run benchmark if requested
    if args.benchmark:
        exporter.benchmark(engine_path)
    
    # Run validation if requested
    if args.validate:
        if not args.val_data:
            raise ValueError("Validation data path required for validation")
        exporter.validate_engine(engine_path, args.val_data)
    
    logger.info("Export complete!")

if __name__ == '__main__':
    main()
