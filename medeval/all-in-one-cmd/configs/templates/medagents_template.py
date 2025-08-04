import argparse
import os
from evalscope import TaskConfig, run_task

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, default='outputs')
args = parser.parse_args()

task_cfg = TaskConfig(
    model=__INFER_SERVED_NAME__,
    api_url=__API_URL_WITH_ENDPOINT__,
    api_key='EMPTY',
    eval_type='service',
    datasets=['general_mcq'],
    dataset_args={
        'general_mcq': {
            'local_path': os.path.join(__INPUT_PATH__, 'medagents'),
            'subset_list': [
                'afrimedqa', 'medbullets', 'medexqa', 'medmcqa',
                'medqa_5options', 'medqa', 'medxpertqa-r', 'medxpertqa-u',
                'mmlu', 'mmlu-pro', 'pubmedqa'
            ],
            'prompt_template': 'Please answer this medical question and select the correct answer\n{query}',
            'query_template': 'Question: {question}\n{choices}\nAnswer: {answer}\n\n',
        }
    },
    work_dir=args.work_dir,
    limit=__LIMIT__,
)
run_task(task_cfg=task_cfg)
