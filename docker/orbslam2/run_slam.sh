#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 <dataset> <sequence> <output_path> <useDisplay>" >&2
  exit 1
fi

DATASET="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
SEQUENCE="$2"
OUTPUT_PATH="$3"
useDisplay="$4"

cd /dpds/ORB_SLAM2

case "$DATASET" in
  "KITTI")
    EXECUTABLE="./Examples/Monocular/mono_kitti"
    VOCAB_FILE="Vocabulary/ORBvoc.txt"
    CONFIG_FILE="Examples/Monocular/KITTI00-02.yaml"   # default; overridden below
    SEQ_PATH="/data/KITTI/$SEQUENCE"
    ;;
  "TUM")
    EXECUTABLE="./Examples/Monocular/mono_tum"
    VOCAB_FILE="Vocabulary/ORBvoc.txt"
    CONFIG_FILE="Examples/Monocular/TUM1.yaml"
    SEQ_PATH="/data/TUM/$SEQUENCE"
    ;;
  "EUROC")
    EXECUTABLE="./Examples/Monocular/mono_euroc"
    VOCAB_FILE="Vocabulary/ORBvoc.txt"
    CONFIG_FILE="Examples/Monocular/EuRoC.yaml"
    SEQ_PATH="/data/EuRoC/$SEQUENCE"
    ;;
  "TARTANAIR")
    EXECUTABLE="./Examples/Monocular/mono_tartanair"
    VOCAB_FILE="Vocabulary/ORBvoc.txt"
    CONFIG_FILE="Examples/Monocular/tartanAir.yaml"
    SEQ_PATH="/data/tartanair/$SEQUENCE"
    ;;
  *)
    echo "Unsupported dataset: $DATASET" >&2
    exit 1
    ;;
esac

if [[ "$DATASET" == "KITTI" ]]; then
  case "$SEQUENCE" in
    "00"|"01"|"02")
      CONFIG_FILE="Examples/Monocular/KITTI00-02.yaml"
      ;;
    "03")
      CONFIG_FILE="Examples/Monocular/KITTI03.yaml"
      ;;
    "04"|"05"|"06"|"07"|"08"|"09"|"10")
      CONFIG_FILE="Examples/Monocular/KITTI04-12.yaml"
      ;;
    *)
      echo "Unsupported KITTI sequence: $SEQUENCE" >&2
      exit 1
      ;;
  esac
fi

mkdir -p "$OUTPUT_PATH"

if [[ ! -d "$SEQ_PATH" ]]; then
  echo "Sequence path not found: $SEQ_PATH" >&2
  exit 2
fi

# small manifest for reproducibility/debugging
{
  echo "dataset=$DATASET"
  echo "sequence=$SEQUENCE"
  echo "executable=$EXECUTABLE"
  echo "vocab=$VOCAB_FILE"
  echo "config=$CONFIG_FILE"
  echo "seq_path=$SEQ_PATH"
  echo "output_path=$OUTPUT_PATH"
  date -Is
} > "$OUTPUT_PATH/run_manifest.txt"

if [[ ! -f "$EXECUTABLE" ]]; then
  echo "Error: Executable '$EXECUTABLE' not found!" >&2
  ls -l Examples/Monocular/ >&2
  exit 3
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Error: Config file '$CONFIG_FILE' not found!" >&2
  exit 4
fi

cmd=("$EXECUTABLE" "$VOCAB_FILE" "$CONFIG_FILE" "$SEQ_PATH" "$useDisplay")
echo "Running: ${cmd[*]}"
"${cmd[@]}" 2>&1 | tee "$OUTPUT_PATH/slam_output.txt"

# Use cp instead of mv to avoid ownership preservation issues with Docker volumes
cp track_thread_poses.txt "$OUTPUT_PATH/track_thread_poses.txt" || true
cp KeyFrameTrajectory.txt "$OUTPUT_PATH/KeyFrameTrajectory.txt" || true

# Clean up the original files
rm -f track_thread_poses.txt KeyFrameTrajectory.txt
