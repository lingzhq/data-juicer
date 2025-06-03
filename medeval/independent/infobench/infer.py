import json
import asyncio
from tqdm import tqdm
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:8901/v1",
    api_key="EMPTY"
)

CONCURRENCY = 16

async def process_one(item):
    query = f"{item['instruction']}\n{item['input']}"
    
    messages = [
        {"role": "system", "content": "Please provide your response to the following instruction"},
        {"role": "user", "content": query}
    ]
    
    try:
        response = await client.chat.completions.create(
            model="qwen25-7b-ckpt",
            messages=messages,
            temperature=0.0,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

async def main():
    input_file = "./infobench/data/infobench.jsonl"
    output_file = "./infobench/data/response.jsonl"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f]
    
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = [None] * len(items)
    
    async def process_with_semaphore(i, item):
        async with semaphore:
            result = await process_one(item)
            item["output"] = result
            return i, item
    
    tasks = []
    for i, item in enumerate(items):
        tasks.append(process_with_semaphore(i, item))
    
    pbar = tqdm(total=len(tasks), desc="Processing")
    for future in asyncio.as_completed(tasks):
        i, result_item = await future
        results[i] = result_item
        pbar.update(1)
    pbar.close()
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    asyncio.run(main())