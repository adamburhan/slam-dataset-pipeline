from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class SLAMOutput:
    trajectory_path: Path
    tracking_info_path: Optional[Path] = None