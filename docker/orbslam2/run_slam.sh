#!/bin/bash
set -e

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <dataset> <sequence> <output_path>"
    exit 1
fi

DATASET=$1
SEQUENCE=$2
OUTPUT_PATH=$3

cd /dpds/ORB_SLAM2

case $DATASET in 
    "KITTI")
        EXECUTABLE="./Examples/Monocular/mono_kitti"
        VOCAB_FILE="Vocabulary/ORBvoc.txt"
        CONFIG_FILE="Examples/KITTI/kitti.yaml"
        SEQ_PATH="/data/KITTI/$SEQUENCE"
        ;;
    "TUM")
        EXECUTABLE="./Examples/Monocular/mono_tum"
        VOCAB_FILE="Vocabulary/ORBvoc.txt"
        CONFIG_FILE="Examples/TUM1/TUM1.yaml"
        SEQ_PATH="/data/TUM/$SEQUENCE"
        ;;
    "EuRoC")
        EXECUTABLE="./Examples/Monocular/mono_euroc"
        VOCAB_FILE="Vocabulary/ORBvoc.txt"
        CONFIG_FILE="Examples/EuRoC/EuRoC.yaml"
        SEQ_PATH="/data/EuRoC/$SEQUENCE"
        ;;
    *)
        echo "Unsupported dataset: $DATASET"
        exit 1
        ;;
esac

if $DATASET == "KITTI"; then
    case $SEQUENCE in
        "00"|"01"|"02")
            CONFIG_FILE="Examples/Monocular/KITTI00-02.yaml"
            ;;
        "03")
            CONFIG_FILE="Examples/Monocular/KITTI03.yaml"
            ;;
        "04"|"05"|"06"|"07"|"08"|"09"|"10")
            CONFIG_FILE="Examples/Monocular/KITTI04-12.yaml"
            ;;
    esac
fi

mkdir -p "$OUTPUT_PATH"
$EXECUTABLE "$VOCAB_FILE" "$CONFIG_FILE" "$SEQ_PATH" "$OUTPUT_PATH" \
    > "$OUTPUT_PATH/slam_output.txt" 2>&1

