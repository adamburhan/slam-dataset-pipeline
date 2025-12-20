from typing import Any
from .SLAMSystem import SLAMSystem
from .ORBSLAMSystem import ORBSLAMSystem

def get_system(config: Any) -> SLAMSystem:
    """
    Factory function to create a SLAMSystem instance from configuration.
    
    Args:
        config: SystemConfig object or dictionary
    """
    name = getattr(config, 'name', config.get('name') if isinstance(config, dict) else None)
    
    if not name:
        raise ValueError("System config must have 'name'")
        
    name = name.lower()
    
    if name == "orbslam2":
        # In the future, we can pass the runtime config here
        return ORBSLAMSystem()
    else:
        raise ValueError(f"Unknown SLAM system: {name}")
