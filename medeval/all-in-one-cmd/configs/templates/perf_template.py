import argparse
import os
from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, default='outputs')
args = parser.parse_args()

task_cfg = Arguments(
    parallel=[1, 100],
    number=[10, 200],
    model=__INFER_SERVED_NAME__,
    url=__API_URL_WITH_ENDPOINT__,
    api='openai',
    dataset='openqa',
    dataset_path=os.path.join(__INPUT_PATH__, 'perf/open_qa.jsonl'),
    temperature=0.9,
    max_tokens=1024,
    min_prompt_length=10,
    max_prompt_length=4096,
    tokenizer_path=__INFER_MODEL_PATH__,
    extra_args={'ignore_eos': True},
    outputs_dir=args.work_dir
)

results = run_perf_benchmark(task_cfg)

