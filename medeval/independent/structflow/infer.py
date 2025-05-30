import json, os, glob, random, sys, argparse
import asyncio
from typing import List, Dict
from tqdm import tqdm
from openai import AsyncOpenAI


user_name = "user"
assistant_name = "assistant"


class Inference():
    def __init__(self, infer_model, in_path, out_dir, max_concurrency):
        self.infer_model = infer_model
        self.out_path = os.path.join(out_dir, f"{infer_model}_infer.json")
        self.in_path = in_path
        self.example_num = 0
        self.max_concurrency = max_concurrency
        self.client = AsyncOpenAI(
            base_url="http://localhost:8901/v1",
            api_key="EMPTY"
        )

    def _load_examples(self, in_path):
        try:
            with open(in_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.example_num = len(data)
            return data
        except Exception as e:
            raise ValueError(f"Dataset error: {str(e)}")

    async def _call_api(self, messages: List[Dict]) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.infer_model,
                messages=messages,
                temperature=0.01,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return ""

    async def _process_single_conv(self, conv_data: List[Dict]) -> List[Dict]:
        for turn_idx in range(len(conv_data)):
            messages = []
            for history_idx in range(turn_idx):
                messages.append({
                    "role": "user",
                    "content": conv_data[history_idx]["user prompt"]
                })
                messages.append({
                    "role": "assistant", 
                    "content": conv_data[history_idx]["assistant answer"]
                })
            
            current_prompt = conv_data[turn_idx]["user prompt"]
            messages.append({
                "role": "user",
                "content": current_prompt
            })

            response = await self._call_api(messages)
            conv_data[turn_idx]["response"] = response
            
            if "assistant answer" not in conv_data[turn_idx]:
                conv_data[turn_idx]["assistant answer"] = response
                
        return conv_data

    async def _process_batch(self, datas: List[Dict]):
        sem = asyncio.Semaphore(self.max_concurrency)
        
        async def process_with_semaphore(data):
            async with sem:
                return await self._process_single_conv(data["whole_conv"])
        
        tasks = []
        for data in datas:
            tasks.append(process_with_semaphore(data))
        
        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
            results.append(await f)
            
        return datas

    def _save_result(self, result):
        try:
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            with open(self.out_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Save failed: {str(e)}")

    async def async_run(self):
        datas = self._load_examples(self.in_path)
        await self._process_batch(datas)
        self._save_result(datas)

    def __call__(self):
        asyncio.run(self.async_run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_model", type=str, default="qwen25-7b-ckpt", help="infer model name")
    parser.add_argument("--in_path", type=str, default="./structflow/data/structflow.json")
    parser.add_argument("--out_dir", type=str, default="./structflow/data")
    parser.add_argument("--max_concurrency", type=int, default=16, help="max")
    args = parser.parse_args()
    
    Inference(
        infer_model=args.infer_model,
        in_path=args.in_path,
        out_dir=args.out_dir,
        max_concurrency=args.max_concurrency
    )()
