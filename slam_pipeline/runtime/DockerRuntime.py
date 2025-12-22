import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from .ContainerRuntime import ContainerRuntime

class DockerRuntime(ContainerRuntime):
    def __init__(self, network: str = "host"):
        self.network = network

    def run(self, 
            image: str, 
            command: List[str], 
            volumes: Dict[Path, str], 
            env: Dict[str, str],
            workdir: Optional[str] = None) -> int:
        
        docker_cmd = ["docker", "run", "--rm"]
        
        # Network
        if self.network:
            docker_cmd.append(f"--net={self.network}")
            
        # Volumes
        for host_path, container_path in volumes.items():
            # Ensure host path is absolute
            abs_host_path = str(Path(host_path).resolve())
            docker_cmd.extend(["-v", f"{abs_host_path}:{container_path}"])
            
        # Environment
        for k, v in env.items():
            docker_cmd.extend(["-e", f"{k}={v}"])
            
        # Workdir
        if workdir:
            docker_cmd.extend(["-w", workdir])
            
        # Image and Command
        docker_cmd.append(image)
        docker_cmd.extend(command)
        
        print(f"Executing Docker command: {' '.join(docker_cmd)}")
        
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("Docker execution failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
        return result.returncode
