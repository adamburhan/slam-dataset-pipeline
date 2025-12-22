from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

class ContainerRuntime(ABC):
    @abstractmethod
    def run(self, 
            image: str, 
            command: List[str], 
            volumes: Dict[Path, str], 
            env: Dict[str, str],
            workdir: Optional[str] = None) -> int:
        """
        Run a command inside a container.
        
        Args:
            image: Name/Path of the image (e.g., 'orbslam2:latest' or 'orbslam2.sif')
            command: List of command arguments to run inside the container
            volumes: Dictionary mapping {host_path: container_path}
            env: Dictionary of environment variables
            workdir: Working directory inside the container
            
        Returns:
            Exit code of the process
        """
        pass
