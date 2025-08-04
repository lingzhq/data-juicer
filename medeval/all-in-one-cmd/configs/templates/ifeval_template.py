import argparse
from evalscope import TaskConfig, run_task

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, default='outputs')
args = parser.parse_args()

task_cfg = TaskConfig(
    model=__INFER_SERVED_NAME__,
    api_url=__API_URL_WITH_ENDPOINT__,
    api_key='EMPTY',
    eval_type='service',
    datasets=['ifeval'],
    work_dir=args.work_dir,
    limit=__LIMIT__,
)
run_task(task_cfg=task_cfg)
