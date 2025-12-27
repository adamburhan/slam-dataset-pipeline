import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from .ContainerRuntime import ContainerRuntime

class SingularityRuntime(ContainerRuntime):
    def __init__(self, gpu: bool = False):
        self.gpu = gpu

    def run(self, 
            image: str, 
            command: List[str], 
            volumes: Dict[Path, str], 
            env: Dict[str, str],
            workdir: Optional[str] = None) -> int:
        
        # Singularity uses 'exec' to run a command in an existing image file (.sif)
        sing_cmd = ["singularity", "exec"]
        
        if self.gpu:
            sing_cmd.append("--nv")
            
        # Volumes (Singularity uses --bind host:container)
        bind_args = []
        for host_path, container_path in volumes.items():
            abs_host_path = str(Path(host_path).resolve())
            bind_args.append(f"{abs_host_path}:{container_path}")
            
        if bind_args:
            sing_cmd.extend(["--bind", ",".join(bind_args)])
            
        # Environment
        # Singularity inherits env by default, but we can force specific ones using --env
        env_args = []
        for k, v in env.items():
            env_args.append(f"{k}={v}")
            
        if env_args:
            sing_cmd.extend(["--env", ",".join(env_args)])
            
        # Workdir
        if workdir:
            sing_cmd.extend(["--pwd", workdir])
            
        # Image and Command
        sing_cmd.append(image)
        sing_cmd.extend(command)
        
        print(f"Executing Singularity command: {' '.join(sing_cmd)}")
        
        # Note: Singularity usually streams output to stdout/stderr directly
        result = subprocess.run(
            sing_cmd,
            check=False
        )
        
        if result.returncode != 0:
            print("Singularity execution failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
        return result.returncode
