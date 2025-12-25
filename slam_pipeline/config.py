from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml
from pathlib import Path

@dataclass
class DatasetConfig:
    name: str
    root_dir: str
    sequences: List[str] = field(default_factory=list)
    domains: Optional[List[str]] = None
    difficulties: Optional[List[str]] = None

@dataclass
class SystemConfig:
    name: str
    docker_image: str
    display: bool = True
    mount_slam: bool = False
    runtime: str = "docker"
    runtime_args: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AssociationConfig:
    max_diff: float = 0.02
    require_unique: bool = True
    assign_gt_frame_ids_to_est: bool = True
    strict: bool = True
    interpolate_gt: bool = False

@dataclass
class LoadingConfig:
    est_format: str
    fill_policy: str
    association: AssociationConfig

@dataclass
class AlignmentConfig:
    method: str = "sim3"
    align_ground_truth: bool = False
    only_scale: bool = False

@dataclass
class OutputConfig:
    save_trajectory: bool = True
    save_plots: bool = True
    save_npz: bool = True
    output_dir: str = "results"

@dataclass
class PipelineConfig:
    loading: LoadingConfig
    alignment: AlignmentConfig
    output: OutputConfig
    metrics: List[Any] = field(default_factory=list)

@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    system: SystemConfig
    pipeline: PipelineConfig

    @classmethod
    def from_yaml(cls, path: Path) -> 'ExperimentConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Helper to recursively instantiate dataclasses
        # Note: This is a simple implementation. For production, libraries like dacite or OmegaConf are better.
        
        dataset_cfg = DatasetConfig(**data['dataset'])
        
        system_data = data['system']
        system_cfg = SystemConfig(
            name=system_data['name'],
            docker_image=system_data.get('docker_image', ''),
            runtime=system_data.get('runtime', 'docker'),
            runtime_args=system_data.get('runtime_args', {})
        )
        
        pipeline_data = data['pipeline']
        
        assoc_data = pipeline_data['loading']['association']
        assoc_cfg = AssociationConfig(**assoc_data)
        
        loading_cfg = LoadingConfig(
            est_format=pipeline_data['loading']['est_format'],
            fill_policy=pipeline_data['loading']['fill_policy'],
            association=assoc_cfg
        )
        
        align_cfg = AlignmentConfig(**pipeline_data['alignment'])
        output_cfg = OutputConfig(**pipeline_data['output'])
        
        pipeline_cfg = PipelineConfig(
            loading=loading_cfg,
            alignment=align_cfg,
            output=output_cfg,
            metrics=pipeline_data.get('metrics', [])
        )
        
        return cls(
            dataset=dataset_cfg,
            system=system_cfg,
            pipeline=pipeline_cfg
        )
