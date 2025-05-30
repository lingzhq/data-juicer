from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen25-7b-ckpt',
    api_url='http://127.0.0.1:8801/v1/chat/completions',
    api_key='EMPTY',
    eval_type='service',
    datasets=['general_mcq'],
    dataset_args={
        'general_mcq': {
            "local_path": "./medeval/med_data/medagents/",
            "subset_list": [
                "afrimedqa", "medbullets", "medexqa",
                "medmcqa", "medqa_5options", "medqa",
                "medxpertqa-r", "medxpertqa-u", "mmlu",
                "mmlu-pro", "pubmedqa"
            ],
            "prompt_template": "Please answer this medical question and select the correct answer\n{query}",
            "query_template": "Question: {question}\n{choices}\nAnswer: {answer}\n\n",
        }
    },
    work_dir='./outputs/medagents'
)
run_task(task_cfg=task_cfg)

