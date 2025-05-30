from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen25-7b-ckpt',
    api_url='http://127.0.0.1:8801/v1/chat/completions',
    api_key='EMPTY',
    eval_type='service',
    datasets=['ifeval'],
    work_dir='./outputs/ifeval'
)
run_task(task_cfg=task_cfg)

