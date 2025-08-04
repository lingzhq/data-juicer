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
    datasets=['general_qa'],
    dataset_args={
        'general_qa': {
            'local_path': os.path.join(__INPUT_PATH__, 'medjourney'),
            'subset_list': [
                'dp', 'dqa', 'dr', 'drg', 'ep', 'hqa', 'iqa', 'mp', 'mqa',
                'pcds', 'pdds', 'tp'
            ],
            'prompt_template': '请回答下述问题\n{query}',
        }
    },
    work_dir=args.work_dir,
    limit=__LIMIT__,
)
run_task(task_cfg=task_cfg)
