import argparse
import sys
import yaml
from loguru import logger
import os

try:
    from med_evaluators import MedEvaluator, DotDict
except ImportError:
    class DotDict(dict):
        __getattr__ = dict.get; __setattr__ = dict.__setitem__; __delattr__ = dict.__delitem__
    from med_evaluators import MedEvaluator

def generate_config_from_template(template_name, output_dir, **kwargs):

    template_path = os.path.join('configs/templates', f'{template_name}_template.py')
    logger.info(f"Generating config for '{template_name}' from template: {template_path}")

    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
    except Exception as e:
        raise IOError(f"Failed to read template file '{template_path}': {e}")

    for key, value in kwargs.items():
        placeholder = f"__{key.upper()}__"
        template_content = template_content.replace(placeholder, repr(value))

    output_config_path = os.path.join(output_dir, f'generated_{template_name}_config.py')
    os.makedirs(os.path.dirname(output_config_path), exist_ok=True)
    
    try:
        with open(output_config_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        logger.success(f"Dynamically generated config saved to: {output_config_path}")
        return output_config_path
    except Exception as e:
        raise IOError(f"Failed to write generated config to '{output_config_path}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Unified command-line runner for MedEval.")
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--infer-model-path', type=str, required=True)
    parser.add_argument('--flames-model-path', type=str, required=True)
    parser.add_argument('--config-template', type=str, default='configs/all_in_one.yaml')
    parser.add_argument('--infer-served-name', type=str, required=True)
    parser.add_argument('--eval-served-name', type=str, required=True)
    parser.add_argument('--infer-port', type=int, default=8901)
    parser.add_argument('--eval-port', type=int, default=8902)
    parser.add_argument('--env-name', type=str, required=True)
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()

    generated_config_dir = 'configs/generated'

    infer_api_url_base = f"http://127.0.0.1:{args.infer_port}/v1"
    
    common_params = {
        'INPUT_PATH': args.input_path,
        'INFER_MODEL_PATH': args.infer_model_path,
        'INFER_SERVED_NAME': args.infer_served_name,
        'LIMIT': args.limit,
    }

    task_specific_params = {
        'ifeval': {'API_URL_WITH_ENDPOINT': f'{infer_api_url_base}/chat/completions'},
        'medagents': {'API_URL_WITH_ENDPOINT': f'{infer_api_url_base}/chat/completions'},
        'medjourney': {'API_URL_WITH_ENDPOINT': f'{infer_api_url_base}/chat/completions'},
        'perf': {'API_URL_WITH_ENDPOINT': f'{infer_api_url_base}/chat/completions'},
    }

    try:
        generated_paths = {}
        for task_name in ['ifeval', 'medagents', 'medjourney', 'perf']:
            all_params = {**common_params, **task_specific_params[task_name]}
            path = generate_config_from_template(task_name, generated_config_dir, **all_params)
            generated_paths[task_name] = path
    except Exception as e:
        logger.error(f"Fatal error during config generation: {e}")
        sys.exit(1)

    with open(args.config_template, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    logger.info("Overwriting template with command-line arguments and generated paths...")
    
    config_dict['input_path'] = args.input_path
    config_dict['output_root_path'] = args.output_path
    config_dict['flames_model_path'] = args.flames_model_path
    config_dict['env_name'] = args.env_name
    
    config_dict['infer_model'] = args.infer_served_name
    config_dict['eval_model'] = args.eval_served_name
    config_dict['infer_api_url'] = infer_api_url_base
    config_dict['eval_api_url'] = f"http://127.0.0.1:{args.eval_port}/v1"
    
    config_dict['generated_config_paths'] = generated_paths
    
    config = DotDict(config_dict)

    try:
        evaluator = MedEvaluator(eval_config=config)
        evaluator.run(eval_type=None, eval_obj=None)
        logger.success("Evaluation pipeline completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
