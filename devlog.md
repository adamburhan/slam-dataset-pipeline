## Implementation TODO List

### PHASE 1: Core Abstractions & Data Flow (SIMPLIFIED - Critical Path)

**Key Insight:** ORB-SLAM2 expects `sequence_dir/image_0/*.png` + `times.txt`. Keep it simple.
We normalize datasets at filesystem level, not in Python. Python just orchestrates.

#### 1.1 Simplify Sequence Dataclass
- [ ] Update `Sequence` to minimal interface:
  ```python
  @dataclass
  class Sequence:
      id: str                          # "00", "01", etc.
      dataset_name: str                # "KITTI"
      sequence_path: Path              # /data/raw/kitti/sequences/00 (contains image_0/ and times.txt)
      ground_truth_file: Optional[Path] = None  # /data/raw/kitti/poses/00.txt
  ```
- [ ] Add `load_ground_truth() -> np.ndarray` method:
  - [ ] Parse KITTI pose format (12 values per line → 3x4 transformation)
  - [ ] Convert to 4x4 homogeneous matrices
  - [ ] Return as numpy array (N, 4, 4)
  - [ ] Return None if no ground truth file

#### 1.2 Implement Minimal Dataset Abstraction
- [ ] Create `slam_pipeline/datasets/Dataset.py` with ABC:
  ```python
  class Dataset(ABC):
      @abstractmethod
      def list_sequences(self) -> List[str]:
          """Return list of sequence IDs (e.g., ['00', '01', '02'])"""
          pass
      
      @abstractmethod
      def get_sequence(self, seq_id: str) -> Sequence:
          """Return Sequence object for given ID"""
          pass
  ```

#### 1.3 Implement KittiDataset (30 lines total)
- [ ] Create `slam_pipeline/datasets/KittiDataset.py`
- [ ] Constructor: `__init__(self, root_path: Path)`
  - [ ] Store `sequences_dir = root / "sequences"`
  - [ ] Store `poses_dir = root / "poses"`
- [ ] Implement `list_sequences()`:
  - [ ] Return `[d.name for d in sequences_dir.iterdir() if d.is_dir()]`
- [ ] Implement `get_sequence(seq_id)`:
  - [ ] Return `Sequence(id=seq_id, dataset_name="KITTI", sequence_path=sequences_dir/seq_id, ground_truth_file=poses_dir/f"{seq_id}.txt")`

#### 1.4 Fix ORBSLAMSystem to Return Trajectory Path
- [ ] Update `ORBSLAMSystem.run()` signature: `def run(self, sequence: Sequence, output_dir: Path) -> Path`
- [ ] Fix volume mounts:
  - [ ] Sequence mount: `-v {sequence.sequence_path}:/data/{sequence.dataset_name}/{sequence.id}`
  - [ ] Output mount: `-v {output_dir}:/output`
  - [ ] Set working dir: `-w /output` (so ORB-SLAM2 writes trajectory there)
- [ ] After subprocess.run():
  - [ ] Check if `output_dir / "KeyFrameTrajectory.txt"` exists
  - [ ] Raise error if not
  - [ ] Return `Path(output_dir / "KeyFrameTrajectory.txt")`
- [ ] Replace print() with logging

#### 1.5 Test End-to-End (45 minutes total)
- [ ] Write simple test in `main.py`:
  ```python
  dataset = KittiDataset("/data/raw/kitti")
  seq = dataset.get_sequence("00")
  slam = ORBSLAMSystem()
  traj_path = slam.run(seq, "/data/outputs/orbslam2/kitti/00")
  gt = seq.load_ground_truth()
  print(f"Estimated trajectory: {traj_path}")
  print(f"Ground truth shape: {gt.shape if gt is not None else 'None'}")
  ```
- [ ] Run and verify trajectory file is created
- [ ] Verify ground truth loads correctly

### PHASE 2: Configuration System

#### 2.1 Setup Configuration Infrastructure
- [ ] Add dependencies: `pyyaml` to `pyproject.toml`
- [ ] Create `slam_pipeline/config/` module
- [ ] Create config loader utility

#### 2.2 Dataset Configurations
- [ ] Update `configs/datasets/kitti.yaml`:
  ```yaml
  name: kitti_odometry
  type: KittiDataset
  root_path: /data/raw/kitti
  ```
- [ ] Create dataset factory: `load_dataset(config_path) -> Dataset`

#### 2.3 SLAM System Configurations  
- [ ] Update `configs/slam/orbslam2.yaml`:
  ```yaml
  name: orb_slam2
  type: ORBSLAMSystem
  docker_image: orbslam2:latest
  singularity_image: /path/to/orbslam2.sif
  runtime: docker  # or singularity
  ```
- [ ] Create SLAM config loader
- [ ] Add runtime mode selection (docker vs singularity)

---

### HANDLING DIFFERENT SLAM SYSTEMS (Design Notes)

**Question:** What if different SLAM systems don't work like ORB-SLAM2?

**Answer:** Use the **Adapter Pattern** - each SLAMSystem implementation handles its own quirks internally.

#### Strategy: Normalize at the Container Boundary

```python
class SLAMSystem(ABC):
    @abstractmethod
    def run(self, sequence: Sequence, output_dir: Path) -> Path:
        """
        Contract: Given a sequence dir (with images + times.txt),
        produce a trajectory file at output_dir.
        
        Implementation can do ANYTHING internally to make this work.
        """
        pass
```

#### Example: Different SLAM Systems

**ORB-SLAM2:**
- Expects: `sequence_dir/image_0/*.png` + `times.txt`
- Outputs: `KeyFrameTrajectory.txt` in CWD
- Solution: Mount sequence dir, set CWD to output dir ✅ (current approach)

**LIO-SAM (expects ROS bags):**
- Expects: ROS bag with `/camera/image` + `/imu/data` topics
- Outputs: Trajectory in custom format
- Solution:
  ```python
  class LIOSAMSystem(SLAMSystem):
      def run(self, sequence: Sequence, output_dir: Path) -> Path:
          # 1. Convert image_0/*.png + times.txt to ROS bag
          bag_path = self._create_bag_from_images(sequence)
          
          # 2. Run LIO-SAM container with bag
          self._run_liosam_container(bag_path, output_dir)
          
          # 3. Convert LIO-SAM output to TUM format
          raw_traj = output_dir / "lio_sam_poses.pcd"
          std_traj = self._convert_to_tum_format(raw_traj)
          
          return std_traj
  ```

**RTAB-Map (expects different directory structure):**
- Expects: `rgb/`, `depth/`, `poses.txt`
- Solution:
  ```python
  class RTABMapSystem(SLAMSystem):
      def run(self, sequence: Sequence, output_dir: Path) -> Path:
          # 1. Create temp dir with RTAB-Map layout
          temp_dir = self._prepare_rtabmap_layout(sequence)
          
          # 2. Run RTAB-Map
          self._run_container(temp_dir, output_dir)
          
          # 3. Return trajectory path
          return output_dir / "trajectory.txt"
  ```

**VINS-Mono (needs config files per dataset):**
- Expects: Config YAML with intrinsics
- Solution:
  ```python
  class VINSMonoSystem(SLAMSystem):
      def __init__(self, config_template: Path):
          self.config_template = config_template
      
      def run(self, sequence: Sequence, output_dir: Path) -> Path:
          # 1. Generate dataset-specific config
          config = self._generate_config(sequence, self.config_template)
          
          # 2. Run with generated config
          ...
  ```

#### Key Principle: "Make It Look Right"

Each SLAM system implementation is responsible for:
1. **Input preparation**: Transform `Sequence` into whatever format SLAM needs
2. **Execution**: Run the containerized SLAM system
3. **Output normalization**: Convert SLAM output to standard trajectory format (e.g., TUM format)

The **rest of the pipeline** (preprocessing, metrics) only sees:
```python
trajectory_path: Path  # Always points to a trajectory file in TUM/standard format
```

#### Recommended Trajectory Format: TUM

```
# timestamp tx ty tz qx qy qz qw
1305031102.175304 0.0 0.0 0.0 0.0 0.0 0.0 1.0
1305031102.275304 0.1 0.0 0.0 0.0 0.0 0.0 1.0
...
```

Why TUM?
- ✅ Standard in SLAM community
- ✅ Easy to parse
- ✅ Includes timestamps (needed for alignment)
- ✅ Quaternion representation (no gimbal lock)

#### Implementation Checklist for New SLAM System

When adding a new SLAM system, implement:

- [ ] Dockerfile that builds the SLAM system
- [ ] Wrapper script (like `run_slam.sh`) if needed
- [ ] `SLAMSystem` subclass with `run()` method that:
  - [ ] Prepares input data in SLAM's expected format
  - [ ] Executes containerized SLAM
  - [ ] Converts output to TUM format
  - [ ] Returns Path to trajectory file
- [ ] Config YAML in `configs/slam/`
- [ ] Test with one sequence to verify end-to-end

#### Future: Abstraction Layers for Complex Cases

If you get many SLAM systems with similar needs, extract common logic:

```python
class ROSBagSLAMSystem(SLAMSystem):
    """Base class for SLAM systems that need ROS bags"""
    
    def run(self, sequence: Sequence, output_dir: Path) -> Path:
        bag_path = self._convert_to_bag(sequence)  # Common logic
        traj_path = self._run_slam_on_bag(bag_path, output_dir)  # Subclass-specific
        return self._standardize_output(traj_path)  # Common logic

class LIOSAMSystem(ROSBagSLAMSystem):
    def _run_slam_on_bag(self, bag_path, output_dir):
        # LIO-SAM specific execution
        ...
```

But **don't build this until you have 2-3 SLAM systems working**. YAGNI principle.

---

### PHASE 3: Pipeline Integration

#### 3.1 Implement Preprocessor
- [ ] Create `slam_pipeline/preprocessors/Preprocessor.py` ABC:
  ```python
  class Preprocessor(ABC):
      @abstractmethod
      def process(self, sequence: Sequence, est_trajectory: np.ndarray) -> np.ndarray:
          pass
  ```
- [ ] Create `ConstantVelocityPreprocessor`:
  - [ ] Detect missing/lost frames (NaN or zero poses)
  - [ ] Interpolate using constant velocity model
  - [ ] Return processed trajectory

#### 3.2 Implement Metrics
- [ ] Create `slam_pipeline/metrics/Metric.py` ABC:
  ```python
  class Metric(ABC):
      @abstractmethod
      def compute(self, gt_trajectory: np.ndarray, est_trajectory: np.ndarray) -> pd.DataFrame:
          pass
  ```
- [ ] Implement `RPEMetric` (Relative Pose Error):
  - [ ] Compute relative transformations over delta frames
  - [ ] Calculate translation and rotation errors
  - [ ] Return per-frame errors as DataFrame
- [ ] Implement `ATEMetric` (Absolute Trajectory Error):
  - [ ] Align trajectories (Umeyama alignment or similar)
  - [ ] Compute point-to-point errors
  - [ ] Return statistics (mean, median, std, rmse)

#### 3.3 Create Pipeline Runner
- [ ] Create `slam_pipeline/pipeline/Runner.py`:
  - [ ] Load experiment config
  - [ ] For each (dataset, slam_system, sequence):
    - [ ] Run SLAM → get trajectory path
    - [ ] Load estimated trajectory
    - [ ] Load ground truth trajectory
    - [ ] Run preprocessors
    - [ ] Compute metrics
    - [ ] Save results to `/data/ml_dataset/labels/`
- [ ] Add progress tracking (tqdm or similar)
- [ ] Add error recovery (skip failed sequences, log errors)

### PHASE 4: CLI Scripts

#### 4.1 Implement `run_slam_cli.py`
- [ ] Parse args: `--dataset`, `--slam-system`, `--sequence-id`, `--output-dir`
- [ ] Load configs
- [ ] Execute SLAM only
- [ ] Save trajectory

#### 4.2 Implement `preprocess_cli.py`
- [ ] Parse args: `--trajectory`, `--preprocessor`, `--output`
- [ ] Load trajectory file
- [ ] Run preprocessing
- [ ] Save processed trajectory

#### 4.3 Implement `compute_metrics_cli.py`
- [ ] Parse args: `--gt`, `--est`, `--metrics`, `--output`
- [ ] Load both trajectories
- [ ] Compute specified metrics
- [ ] Save results as CSV

#### 4.4 Implement `run_experiment_cli.py`
- [ ] Parse args: `--config`, `--sequence-id` (optional, for SLURM arrays)
- [ ] Load experiment config
- [ ] Execute full pipeline
- [ ] Generate final ML dataset

### PHASE 5: Production Readiness

#### 5.1 Docker/Singularity Runtime Switching
- [ ] Add runtime detection in `ORBSLAMSystem`
- [ ] Abstract container execution:
  ```python
  def _run_container(self, image, mounts, command):
      if self.runtime == "docker":
          return self._run_docker(...)
      elif self.runtime == "singularity":
          return self._run_singularity(...)
  ```
- [ ] Test both modes

#### 5.2 Output Standardization
- [ ] Define trajectory file format (TUM format recommended):
  ```
  timestamp tx ty tz qx qy qz qw
  ```
- [ ] Update ORB-SLAM2 output to write this format
- [ ] Add trajectory format converters (KITTI → TUM, etc.)

#### 5.3 Path Handling
- [ ] Remove all hardcoded paths
- [ ] Use environment variables or config for:
  - [ ] `DATA_ROOT` (e.g., `/data` or `/scratch/$USER/data`)
  - [ ] `CONTAINER_ROOT` (where .sif files live)
- [ ] Make all paths configurable in YAML

#### 5.4 Logging & Debugging
- [ ] Replace all `print()` with proper logging
- [ ] Add log levels (DEBUG, INFO, WARNING, ERROR)
- [ ] Save logs to file alongside outputs
- [ ] Add `--verbose` and `--debug` flags to CLIs

### PHASE 6: Testing & Validation

#### 6.1 Unit Tests
- [ ] Test `Sequence.get_image_paths()` with mock data
- [ ] Test `KittiDataset.list_sequences()`
- [ ] Test `ConstantVelocityPreprocessor` with synthetic data
- [ ] Test metrics with known ground truth

#### 6.2 Integration Test
- [ ] End-to-end test on KITTI sequence 00:
  - [ ] Run ORB-SLAM2
  - [ ] Preprocess trajectory
  - [ ] Compute RPE/ATE
  - [ ] Verify output files exist and are valid

#### 6.3 Cluster Test
- [ ] Convert Docker image to Singularity
- [ ] Test on compute cluster with SLURM
- [ ] Run array job over multiple sequences
- [ ] Verify all outputs are saved correctly

---

## Quick Start Checklist (Minimal Viable Pipeline)

To get ONE sequence working end-to-end, focus on:

1. ✅ Fix `ORBSLAMSystem.run()` to return trajectory path
2. ✅ Fix volume mount path bug
3. ✅ Implement `KittiDataset` class
4. ✅ Make `Sequence` load ground truth
5. ✅ Implement ONE metric (RPE or ATE)
6. ✅ Wire it together in `main.py` or simple runner script
7. ✅ Test on KITTI 00

**Success criteria**: Running `python main.py` produces:
- Estimated trajectory file
- Ground truth trajectory loaded
- Metric CSV saved to disk

---

## Old Considerations (Archived)
- Do we definitely want ros bags as input always? It might be simpler to have the option of bags vs standard image directories.
    - For kitti, tartanAir, ETH3D it seems easier now to just define a new mono_{dataset}.cc with an added argument for the output_dir (for the estimated trajectories). Will definitely need to think about designing the ros version though since it seems like a simpler option for data logging (maybe not 100% necessary?).

- Is it worth having a Sequence class? This can have as attributes the image directory or bag file, the timestamps, the intrinsics/extrinsics and the groundtruth trajectory. Maybe we can have a different class for a sequence using image directories and timestamps file and directly rosbags?

- SLAMSystem should have a mode for docker vs singularity
- running slam and logging tracking pose almost works, do not forget to explain how we currently have poses for lost tracking as well
- next steps are the actual definition of dataset classes, how they get groundtruth and the actual metrics module