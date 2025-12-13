from abc import ABC, abstractmethod
from typing import List, Optional
from slam_pipeline.datasets.Sequence import Sequence
import numpy as np

class Dataset(ABC):
    @abstractmethod
    def list_sequences(self) -> List[str]:
        """Return list of sequence IDs (e.g, ['00', '01', '02'])"""
        pass
    
    @abstractmethod
    def get_sequence(self, sequence_id: str) -> Sequence:
        """Return Sequence object for given sequence ID"""
        pass
    
    @abstractmethod
    def load_ground_truth(self, sequence: Sequence) -> Optional[np.ndarray]:
        """Load ground truth for a sequence in this dataset's format"""
        pass