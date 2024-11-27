import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Union, Optional

class ModelPruner:
    """Handles model pruning for network optimization"""
    
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.pruning_methods = {
            'l1_unstructured': prune.l1_unstructured,
            'random_unstructured': prune.random_unstructured,
            'ln_structured': prune.ln_structured
        }
    
    def apply_pruning(self, method: str = 'l1_unstructured',
                     amount: float = 0.3,
                     layers_to_prune: Optional[List[str]] = None):
        """Apply pruning to specified layers"""
        if layers_to_prune is None:
            # Default to pruning all Conv2d and Linear layers
            layers_to_prune = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    layers_to_prune.append(name)
        
        for name, module in self.model.named_modules():
            if name in layers_to_prune:
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    self.pruning_methods[method](
                        module,
                        name='weight',
                        amount=amount
                    )
    
    def remove_pruning(self):
        """Remove pruning reparametrization"""
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.remove(module, 'weight')
    
    def get_sparsity_stats(self) -> Dict[str, Dict[str, float]]:
        """Get layer-wise sparsity statistics"""
        stats = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight
                total_params = weight.nelement()
                zero_params = (weight == 0).sum().item()
                sparsity = 100. * zero_params / total_params
                
                stats[name] = {
                    'total_params': total_params,
                    'zero_params': zero_params,
                    'sparsity_percentage': sparsity
                }
        return stats
    
    def iterative_pruning(self, method: str = 'l1_unstructured',
                         initial_amount: float = 0.1,
                         final_amount: float = 0.7,
                         steps: int = 5,
                         eval_fn: Optional[callable] = None):
        """Gradually prune the model while monitoring performance"""
        results = []
        step_size = (final_amount - initial_amount) / steps
        
        for step in range(steps):
            current_amount = initial_amount + step * step_size
            
            # Apply pruning
            self.apply_pruning(method=method, amount=current_amount)
            
            # Evaluate if evaluation function provided
            eval_metric = None
            if eval_fn is not None:
                eval_metric = eval_fn(self.model)
            
            # Get sparsity statistics
            sparsity_stats = self.get_sparsity_stats()
            
            results.append({
                'step': step,
                'pruning_amount': current_amount,
                'eval_metric': eval_metric,
                'sparsity_stats': sparsity_stats
            })
        
        return results
    
    def structured_channel_pruning(self, amount: float = 0.3):
        """Apply structured pruning to reduce number of channels"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=amount,
                    n=2,
                    dim=0  # Prune output channels
                )
