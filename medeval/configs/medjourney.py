from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen25-7b-ckpt',
    api_url='http://127.0.0.1:8801/v1/chat/completions',
    api_key='EMPTY',
    eval_type='service',
    datasets=['general_qa'],
    dataset_args={
        'general_qa': {
            "local_path": "./medeval/med_data/medjourney",
            "subset_list": [
                "dp", "dqa", "dr", "drg", "ep", "hqa", "iqa", "mp", "mqa", "pcds", "pdds", "tp"
            ],
            "prompt_template": "请回答下述问题\n{query}",
        }
    },
    work_dir='./outputs/medjourney'
)
run_task(task_cfg=task_cfg)

