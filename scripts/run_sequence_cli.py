import argparse
from pathlib import Path
from slam_pipeline.config import ExperimentConfig
from slam_pipeline.pipeline.Pipeline import Pipeline

def main():
    parser = argparse.ArgumentParser(description="Run SLAM pipeline on a sequence")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--sequence_id", type=str, required=True, help="Sequence ID to process")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    # Load config using the class method we defined
    config = ExperimentConfig.from_yaml(config_path)
    
    # Initialize and run pipeline
    pipeline = Pipeline(config)
    pipeline.run_sequence(args.sequence_id)

if __name__ == "__main__":
    main()