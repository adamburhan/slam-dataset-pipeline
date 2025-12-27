""" 
See dataset documentation for details on TartanAir dataset structure, sampling, etc.
https://tartanair.org/modalities.html#
"""
from slam_pipeline.datasets.Dataset import Dataset
from slam_pipeline.datasets.Sequence import Sequence
from slam_pipeline.trajectories.trajectory import Trajectory
from slam_pipeline.utils.transformations import pos_quats2SE_matrices
from pathlib import Path
from typing import Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class TartanAirSequence(Sequence):
    """Extended sequence with TartanAir-specific metadata."""
    domain: str = "" # e.g, "abandonedfactory"
    difficulty: str = "" # e.g, "Easy" or "Hard"


class TartanAirDataset(Dataset):
    """
    TartanAir dataset structure:
    root_dir/
    |--- abandonedfactory/
    |    |--- Easy/
    |    |    |--- P000/
    |    |    |--- P001/
    |    |    |___ ...
    |    |___ Hard/
    |         |--- P000/
    |         |___ ...
    |--- hospital/
    |    |--- Easy/
    |    |___ Hard/
    |___ ...
    """
    def __init__(self, root_dir: Path, domains: Optional[list[str]] = None, difficulties: Optional[list[str]] = None):
        self.root_dir = root_dir
        self.domains = domains # Filter to specific domains, None = all
        self.difficulties = difficulties or ["Easy", "Hard"]
        
    def _discover_sequences(self) -> list[tuple[str, str, str]]:
        """Discover all (domain, difficulty, seq_id) tuples."""
        sequences = []
        
        for domain_dir in self.root_dir.iterdir():
            if not domain_dir.is_dir():
                continue
            if self.domains is not None and domain_dir.name not in self.domains:
                continue
            for difficulty in self.difficulties:
                diff_dir = domain_dir / difficulty
                if not diff_dir.exists():
                    continue
                
                for seq_dir in diff_dir.iterdir():
                    if seq_dir.is_dir():
                        sequences.append((domain_dir.name, difficulty, seq_dir.name))
        return sequences
    
    def list_sequences(self):
        """Return flat list of sequence IDs in format 'domain/difficulty/seq_id'."""
        return [f"{d}/{diff}/{seq_id}" for d, diff, seq_id in self._discover_sequences()]
    
    def _parse_sequence_id(self, sequence_id: str) -> tuple[str, str, str]:
        """Parse 'domain/difficulty/seq_id' into components."""
        parts = sequence_id.split("/")
        if len(parts) != 3:
            raise ValueError(f"Invalid sequence_id format: {sequence_id}")
        return tuple(parts)
    
    def get_sequence(self, sequence_id: str) -> TartanAirSequence:
        domain, difficulty, seq_id = self._parse_sequence_id(sequence_id)
        seq_dir = self.root_dir / domain / difficulty / seq_id
        if not seq_dir.exists():
            raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")
        
        return TartanAirSequence(
            id=sequence_id,
            dataset_name="tartanair",
            sequence_dir=seq_dir,
            dataset_root_dir=self.root_dir,
            ground_truth_file=seq_dir / "groundtruth.txt",
            timestamps_file=seq_dir / "times.txt",
            domain=domain,
            difficulty=difficulty,
        )

    def load_ground_truth(self, sequence: TartanAirSequence) -> Trajectory:
        """
        Load TartanAir ground truth poses.

        Each line contains 8 values representing a pose in TUM format (timestamp tx ty tz qx qy qz qw).
        Returns a Trajectory with timestamps and SE(3) poses.
        """
        gt_file = sequence.ground_truth_file
        if not gt_file.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_file}")

        # Load poses
        poses = []
        with open(gt_file, "r") as f:
            for i, line in enumerate(f):
                values = np.fromstring(line, sep=" ")
                if values.size != 8:
                    raise ValueError(
                        f"Line {i} in {gt_file} has {values.size} values, expected 8"
                    )
                poses.append(values[1:])  # Exclude timestamp

        #poses = np.stack(poses, axis=0).astype(np.float64)
        poses = pos_quats2SE_matrices(np.stack(poses, axis=0).astype(np.float64))  # (N,7) -> (N,4,4)

        # Load timestamps
        timestamps_file = sequence.timestamps_file
        if not timestamps_file.exists():
            raise FileNotFoundError(f"Timestamps file not found: {timestamps_file}")

        stamps = np.loadtxt(timestamps_file, dtype=np.float64)
        stamps = np.atleast_1d(stamps)

        if len(stamps) != poses.shape[0]:
            raise ValueError(
                f"GT timestamps ({len(stamps)}) and poses ({poses.shape[0]}) length mismatch"
            )

        frame_ids = np.arange(len(stamps), dtype=np.int32)

        return Trajectory(
            stamps=stamps,
            poses=poses,
            frame_ids=frame_ids,
        )
        
    def list_domains(self) -> list[str]:
        """List all available domains."""
        return list(set(d for d, _, _ in self._discover_sequences()))
    
    def list_by_domain(self, domain: str) -> list[str]:
        """List sequences for a specific domain."""
        return [f"{d}/{diff}/{s}" for d, diff, s in self._discover_sequences() if d == domain]
    
    def list_by_difficulty(self, difficulty: str) -> list[str]:
        """List sequences for a specific difficulty."""
        return [f"{d}/{diff}/{s}" for d, diff, s in self._discover_sequences() if diff == difficulty]