from abc import ABC, abstractmethod
from pathlib import Path
from slam_pipeline.slam_systems.common import SLAMOutput

class SLAMSystem(ABC):
    @abstractmethod
    def run(self, sequence, output_dir: Path) -> SLAMOutput:
        """
        Runs SLAM on the given sequence.

        Returns the path to an estimated trajectory file.
        """
        pass