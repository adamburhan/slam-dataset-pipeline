from abc import ABC, abstractmethod
from pathlib import Path

class SLAMSystem(ABC):
    @abstractmethod
    def run(self, sequence, output_dir: Path) -> Path:
        """
        Runs SLAM on the given sequence.

        Returns the path to an estimated trajectory file.
        """
        pass