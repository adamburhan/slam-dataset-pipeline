from typing import Any
from .SLAMSystem import SLAMSystem
from .ORBSLAMSystem import ORBSLAMSystem
from slam_pipeline.runtime.DockerRuntime import DockerRuntime
from slam_pipeline.runtime.SingularityRuntime import SingularityRuntime

def get_system(config: Any) -> SLAMSystem:
    """
    Factory function to create a SLAMSystem instance from configuration.
    
    Args:
        config: SystemConfig object or dictionary
    """
    name = getattr(config, 'name', config.get('name') if isinstance(config, dict) else None)
    runtime_type = getattr(config, 'runtime', config.get('runtime', 'docker') if isinstance(config, dict) else 'docker')
    
    if not name:
        raise ValueError("System config must have 'name'")
        
    name = name.lower()
    
    # Create Runtime
    if runtime_type == "docker":
        runtime = DockerRuntime()
    elif runtime_type == "singularity":
        runtime = SingularityRuntime()
    else:
        raise ValueError(f"Unknown runtime: {runtime_type}")
    
    if name == "orbslam2":
        return ORBSLAMSystem(config, runtime)
    else:
        raise ValueError(f"Unknown SLAM system: {name}")
