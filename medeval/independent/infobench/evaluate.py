import json
import os
import asyncio
import argparse
from os.path import join, exists
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from typing import List, Dict

SYS_MSG ="Based on the provided Input (if any) and Generated Text, answer the ensuing Questions with either a YES or NO choice. Your selection should be based on your judgment as well as the following rules:\n\n- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. However, note that even minor inaccuracies exclude the text from receiving a 'YES' rating. As an illustration. consider a question that asks. \"Does each sentence in the generated text use a second person?” If even one sentence does not use the second person, the answer should NOT be 'YES'. To qualify for a 'YES' rating, the generated text must be entirely accurate and relevant to the question\n\n- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question. For instance, if the question asks. \"Is the second sentence in the generated text a compound sentence?\" and the generated text only has one sentence. it offers no relevant information to answer the question. Consequently, the answer should be 'NO'.'''"


def load_jsonl(file_path):
    "General function to load jsonl file"
    _data = []
    with open(file_path, 'r') as f:
        for data in f:
            jline = json.loads(data)
            _data.append(jline)
    return _data


def bool_ratio(fpath):
    "Calculate true false ratio for eval results"
    _data = load_jsonl(fpath)
    count = {"true":0, "false":0}
    for entry in _data:
        if entry.get("eval", None) is None:
            print("Wrong output")
            print(entry['id'])
        if len(entry['decomposed_questions']) != len(entry['eval']):
            print("Wrong length")
            print(entry['id'])
        if None in entry['eval']:
            print("None in eval")
            print(entry['id'])
        
        for eva_value in entry['eval']:
            if eva_value:
                count["true"] += 1
            else:
                count["false"] += 1
    
    print("-------- True False Table --------")
    print(count)
    print(f"Percentage of True: {count['true']/sum(count.values())}")
    return

    
async def process_question(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    question: str,
    input_task: str,
    output: str,
    eval_model: str,
    temperature: float
) -> str:
    async with sem:
        retry_count = 0
        max_retries = 3
        message = []
        if input_task:
            content = f"{SYS_MSG}\n\nInput:\n\"{input_task}\"\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
        else:
            content = f"{SYS_MSG}\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
        
        message.append({"role": "user", "content": content})
        
        while retry_count < max_retries:
            try:
                completion = await client.chat.completions.create(
                    model=eval_model,
                    messages=message,
                    temperature=temperature,
                )
                generation = completion.choices[0].message.content

                return parse_response(generation)

            except Exception as e:
                print(f"Error: {e}, retrying... ({retry_count+1}/{max_retries})")
                retry_count += 1
                await asyncio.sleep(5)
        return "None"


def parse_response(generation: str) -> str:
    if "</think>" in generation:
        response_part = generation.split("</think>")[-1].strip()
    else:
        response_part = generation
    
    clean_gen = response_part.lower().strip()
    
    if any(word in clean_gen for word in ["yes", "correct", "true", "affirmative"]):
        if "no" not in clean_gen and "not" not in clean_gen:
            return "Yes"
    
    if any(word in clean_gen for word in ["no", "incorrect", "false"]):
        if "yes" not in clean_gen and "correct" not in clean_gen:
            return "No"
    
    if clean_gen.startswith(("yes", "y")):
        return "Yes"
    if clean_gen.startswith(("no", "n")):
        return "No"
    
    return "None"


async def process_entry(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    entry: Dict,
    eval_model: str,
    temperature: float
) -> Dict:
    if entry.get('eval', None) is not None:
        return entry
    
    input_task = entry['input']
    output = entry['output']
    if output is None:
        return entry
    
    tasks = [
        process_question(client, sem, q, input_task, output, eval_model, temperature)
        for q in entry['decomposed_questions']
    ]
    
    results = await asyncio.gather(*tasks)
    
    bool_results = []
    for res in results:
        if res == "Yes":
            bool_results.append(True)
        elif res == "No":
            bool_results.append(False)
        else:
            bool_results.append(None)
    
    entry['eval'] = bool_results
    return entry

async def run_evaluation(client, in_path, o_dir, eval_model="qwen3-32b", temperature=0.0):
    _data = load_jsonl(in_path)
    _model_name = in_path.split('/')[1].split('_')[0]
    
    _o_dir = join(o_dir, eval_model)
    os.makedirs(_o_dir, exist_ok=True)
    
    _opath = join(_o_dir, f"{_model_name}_DecomposeEval.jsonl")
    
    existing_data = {}
    if os.path.exists(_opath):
        _exist = load_jsonl(_opath)
        existing_data = {item['id']: item for item in _exist}
    
    sem = asyncio.Semaphore(10)
    
    tasks = []
    for entry in _data:
        if entry['id'] in existing_data:
            continue
        tasks.append(process_entry(client, sem, entry, eval_model, temperature))
    
    results = []
    pbar = tqdm(total=len(tasks), desc="Processing entries")
    for batch in asyncio.as_completed(tasks):
        result = await batch
        results.append(result)
        pbar.update(1)
    pbar.close()
    
    final_data = []
    for entry in _data:
        if entry['id'] in existing_data:
            final_data.append(existing_data[entry['id']])
        else:
            final_data.append(next(r for r in results if r['id'] == entry['id']))
    
    with open(_opath, 'w') as f:
        for item in final_data:
            f.write(json.dumps(item) + '\n')
    
    bool_ratio(_opath)
    return _opath

async def main_run(args):
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=args.base_url,
        timeout=60.0
    )
    
    await run_evaluation(
        client,
        args.input,
        args.output_dir,
        args.model,
        args.temperature
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--model", type=str, default="qwen3-32b", help="model name to be used for evaluation")
    
    parser.add_argument("--input", type=str, default="./infobench/data/response.jsonl", help="path to the results file")
    parser.add_argument("--output_dir", type=str, default="./infobench/data", help="path to the output folder")
    
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature to be used for evaluation")
    parser.add_argument("--base_url", type=str, default="http://0.0.0.0:8902/v1", help="Local API server address")
    args = parser.parse_args()
    asyncio.run(main_run(args))