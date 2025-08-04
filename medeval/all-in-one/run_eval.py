import argparse
import sys
import yaml
from loguru import logger

from med_evaluators import MedEvaluator

def main():
    parser = argparse.ArgumentParser(description="Standalone Medical Evaluator Script.")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    config_path = args.config
    logger.info(f"Loading configuration from: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load or parse YAML config: {e}")
        sys.exit(1)

    if not isinstance(config, dict):
        logger.error("Configuration file content is not a valid dictionary.")
        sys.exit(1)
        
    logger.info("Configuration loaded successfully.")
    
    try:
        evaluator = MedEvaluator(eval_config=config)
    except Exception as e:
        logger.error(f"Failed to initialize MedEvaluator: {e}")
        sys.exit(1)

    logger.info(f"Starting evaluation for task: {config.get('med_task', 'N/A')}")
    try:
        results = evaluator.run(eval_type=None, eval_obj=None)
        
        logger.success("Evaluation completed.")
        if results:
            logger.info("Final results:")
            logger.info(results)

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
