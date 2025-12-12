# SLAM Dataset Generation Pipeline

This repository provides a reproducible pipeline to generate machine learning datasets from SLAM systems.

The core idea:

> Given a dataset (e.g., KITTI) and a SLAM system (e.g., ORB-SLAM2), run SLAM on each sequence, preprocess trajectories, compute pose error metrics (RPE, ATE), and export ML-ready labels.

---

## Goals

- **Reproducible**: The same Docker image runs locally and on the compute cluster (after converting to singularity).
- **Extensible**:
  - Add new datasets via dataset adapters.
  - Add new SLAM systems via a common interface.
  - Add new metrics without touching core pipeline logic.
- **Separation of concerns**:
  - SLAM execution, data loading, preprocessing and metric computation are clearly separated.
- **Cluster-friendly**:
  - Easy to run as SLURM array jobs: one job = one `(dataset, slam_system, sequence)`.

---

## High-Level Architecture

The pipeline is built around four main abstractions:

1. **Dataset**
2. **SlamSystem**
3. **Preprocessor**
4. **Metric**

### Dataset

Represents a dataset in a **logical, uniform way**, regardless of its actual on-disk layout.

A `Dataset` exposes (conceptually):

```python
class Dataset(ABC):
    name: str

    @abstractmethod
    def list_sequences(self) -> List[str]:
        ...

    @abstractmethod
    def get_sequence(self, seq_id: str) -> "Sequence":
        ...
```

A `Sequence` gives access to:
- image paths (mono/stereo/RGB-D) or other sensor data (imu, LiDAR,...)
- timestamps
- intrinsics/extrinsics
- ground-truth poses

Concrete implementations live in `slam_pipeline/datasets/`:
- KittiDataset
- EurocDataset
- TartanAirDataset
- etc.
Each adapter knows how to read the original dataset layout and present it through a standard interface the rest of the pipeline understands.

### SlamSystem
Represents a SLAM algorithm that ca be run on a `Sequence`.
Conceptually:
```python
class SlamSystem(ABC):
    name: str

    @abstractmethod
    def run(self, sequence: Sequence, output_dir: Path) -> Path:
        """
        Runs SLAM on the given sequence.

        Returns the path to an estimated trajectory file.
        """
        ...
```
Concrete implementations wrap the actual binaries and configs:
- `ORBSLAM2System`
- `LOAMSystem`
- any other SLAM system that can be invoked from the command line.

A `SlamSystem` implementation is responsible for:
- Assembling the correct command line.
- Using dataset-specific paths/resources (e.g, KITTI-calibrated yaml).
- Writing outputs into a known location (`output_dir`).
SLAM systems are configured via YAML files in `configs/slam`.

### Preprocessor
Preprocessing occurs **after SLAM output** but **before metric computation**.
Examples:
- Constant velocity model to fill missing poses (tracking loss)
- Pose smoothing or interpolation
- Timestamp alignment
```python
class Preprocessor(ABC):
    def process(self, sequence, est_traj) -> processed_traj:
        ...
```

### Metric
Metrics operate on ground-truth + estimated trajectories.
Examples:
- Relative Pose Error (RPE)
- Absolute Trajectory Error (ATE)
```python
class Metric(ABC):
    def compute(self, gt, est):
        ...
```
Every metric returns ML-ready data, typically saved as CSV.

---

## Pipeline Flow
The full dataset-generation workflow is:
```mathematica
Dataset -> SLAM -> Preprocessing -> Metrics -> ML labels
```
---

## Experiments
An **experiment** is a YAML file describing a full batch of dataset-generation tasks.
Example:
```yaml
dataset: kitti_odometry
sequences: [00, 01, 02]

slam_systems:
  - orb_slam2

preprocessing:
  - constant_velocity

metrics:
  - rpe

stages:
  - run_slam
  - preprocess
  - compute_metrics

output_root: /data
```
Run it via:
```bash
python scripts/run_experiment_cli.py --config configs/experiments/kitti_orbslam2.yaml
```

---

## Data Layout and Paths
We assume the host filesystem uses:
```bash
/data/
  raw/
    kitti/
      sequences/
      poses/
  outputs/
    orb_slam2/kitti_odometry/00/
  ml_dataset/
    labels/
      orb_slam2/kitti_odometry/00.csv
```
Containers must mount `/data` so that:
- SLAM reads `/data/raw/...`
- SLAM writes `/data/outputs/...`
- Pipeline writes  `/data/ml_dataset/...`
---

## Container Workflow
1. Build docker image
```bash
docker build -t slam_orbslam2 docker/orbslam2/
```
2. Convert to Singularity on the cluster
```bash
sudo APPTAINER_NOHTTPS=1 apptainer build slam_orbslam2.sif docker-daemon://slam_orbslam2:latest
sudo chmod 777 slam_orbslam2.sif
```
3. Reference the image in YAML
```yaml
sif_path: /project/$USER/containers/slam_orbslam2.sif
```
4. Python spawns the SLAM container
Python does not run SLAM itself. Instead, OrbSlam2System.run() calls
```bash
singularity exec --bind /data:/data slam_orbslam2.sif bash run_slam.sh ... 
```
---

## Running on a compute cluster (SLURM)
**Example SLURM script**
```bash
#!/bin/bash
#SBATCH --job-name=slam_dataset
#SBATCH --array=0-10
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

module load python
module load singularity

source /project/$USER/venvs/slam_pipeline/bin/activate

SEQ=${SLURM_ARRAY_TASK_ID}

python scripts/run_experiment_cli.py \
  --config configs/experiments/kitti_orbslam2.yaml \
  --sequence-id $SEQ
```
This allows:
- full dataset processing
- parallelism over sequences
- cluster scalability
---

## Preprocessing and metrics lifecycle
After SLAM:
`Preprocessor.process()` receives:
```nginx
sequence
raw_estimated_trajectory
```
It outputs:
```nginx
processed_trajectory
```

Which gets passed to metrics.

---

## Future extension: SLAM internal signals as features

In future work, SLAM systems may be instrumented (via ROS) to export richer per-frame diagnostics such as tracking state, keypoint statistics, residual distributions and optimizer costs.
These will be written alongside the trajectory in the `outputs/ ` directory and consumed by a separate `FeatureExtractor` module, which builds ML features from both sensor data and SLAM-internal signals.

The current architecture (Dataset, SlamSystem, Preprocessor, Metric) is designed to be compatible with this extension without major changes to the pipeline.