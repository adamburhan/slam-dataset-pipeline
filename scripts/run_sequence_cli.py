import argparse
from pathlib import Path
from slam_pipeline.config import ExperimentConfig
from slam_pipeline.pipeline.Pipeline import Pipeline
import yaml

def main():
    parser = argparse.ArgumentParser(description="Run SLAM pipeline on a sequence")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--sequence_id", type=str, required=True, help="Sequence ID to process")
    parser.add_argument("--dataset_root", type=str, help="Root directory of the dataset")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    # Load config using the class method we defined
    config = ExperimentConfig.from_yaml(config_path)

    print("display =", config.system.display)
    print("system_data keys in yaml =", yaml.safe_load(open(args.config))["system"].keys())


    if args.dataset_root:
        config.dataset.root_dir = Path(args.dataset_root)
        print("root_dir:", config.dataset.root_dir)

    if args.output_dir:
        config.pipeline.output.output_dir = Path(args.output_dir)
        print("output_dir:", config.pipeline.output.output_dir)
    
    # Initialize and run pipeline
    pipeline = Pipeline(config)
    pipeline.run_sequence(args.sequence_id)

if __name__ == "__main__":
    main()