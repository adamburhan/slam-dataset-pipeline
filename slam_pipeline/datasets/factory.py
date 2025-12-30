from pathlib import Path
from typing import Dict, Any
from .Dataset import Dataset
from slam_pipeline.datasets.KittiDataset import KittiDataset
from slam_pipeline.datasets.TartanAirDataset import TartanAirDataset
from slam_pipeline.datasets.EurocDataset import EurocDataset
# from .TartanAirDataset import TartanAirDataset # Example

def get_dataset(config: Any) -> Dataset:
    """
    Factory function to create a Dataset instance from configuration.
    
    Args:
        config: DatasetConfig object or dictionary with 'name' and 'root_dir'
    """
    # Handle both dataclass and dict
    name = getattr(config, 'name', config.get('name') if isinstance(config, dict) else None)
    root_dir = getattr(config, 'root_dir', config.get('root_dir') if isinstance(config, dict) else None)
    
    if not name or not root_dir:
        raise ValueError("Dataset config must have 'name' and 'root_dir'")
        
    name = name.lower()
    root_path = Path(root_dir)
    
    if name == "kitti":
        return KittiDataset(root_path)
    elif name == "tartanair":
        domains = getattr(config, 'domains', config.get('domains') if isinstance(config, dict) else None)
        difficulties = getattr(config, 'difficulties', config.get('difficulties') if isinstance(config, dict) else None)
        return TartanAirDataset(root_path, domains=domains, difficulties=difficulties)
    elif name == "euroc":
        return EurocDataset(root_path)
    else:
        raise ValueError(f"Unknown dataset: {name}")
