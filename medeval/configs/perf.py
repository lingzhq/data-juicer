from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.arguments import Arguments

task_cfg = Arguments(
    parallel=[1, 10, 50, 100],
    number=[10, 20, 100, 200],
    model='qwen25-7b-ckpt',
    url='http://127.0.0.1:8801/v1/chat/completions',
    api='openai',
    dataset='random',
    min_tokens=1024,
    max_tokens=1024,
    prefix_length=0,
    min_prompt_length=1024,
    max_prompt_length=1024,
    tokenizer_path='YOUR_EVAL_LLM',
    extra_args={'ignore_eos': True},
    outputs_dir='./outputs/perf'
)
results = run_perf_benchmark(task_cfg)