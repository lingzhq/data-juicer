export VLLM_USE_MODELSCOPE=True
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model YOUR_EVAL_LLM \
  --served-model-name qwen25-7b-ckpt \
  --trust_remote_code \
  --tensor-parallel-size 2 \
  --port 8901 \