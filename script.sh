#!/bin/bash

# nvbug/5387694 script
# Tested on 8xH100s

# Just used this easy model that I know would fit.
# I changed some of the config parameters to avoid any kind of OOM issues but I think you could definitely use a larger model.
MODEL_PATH=/home/scratch.trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct-FP8

export UCX_TLS=^cuda_ipc
export PATH=$HOME/.local/bin:$PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LLM_MODELS_ROOT=/home/scratch.trt_llm_data/llm-models/

cat > ctx.yml << EOF
disable_overlap_scheduler: True
cache_transceiver_config:
  backend: ucx
print_iter_log: true
EOF

cat > gen.yml << EOF
print_iter_log: true
cache_transceiver_config:
  backend: ucx
EOF

cat > disagg-config.yml << EOF
hostname: localhost
port: 36125
backend: pytorch
context_servers:
  num_instances: 1
  urls:
      - "localhost:36123"
generation_servers:
  num_instances: 1
  urls:
      - "localhost:36124"
EOF

# CTX side first - Pipeline parallelism across 4 GPUs
CUDA_VISIBLE_DEVICES=0,1 trtllm-serve \
  "$MODEL_PATH" \
  --host localhost \
  --port 36123 \
  --backend pytorch \
  --tp_size 1 \
  --ep_size 1 \
  --pp_size 2 \
  --log_level trace \
  --extra_llm_api_options ./ctx.yml &> log_ctx_0 &

# Wait for context server to start
sleep 10

# GEN side next - Tensor parallelism across 4 GPUs
CUDA_VISIBLE_DEVICES=2 trtllm-serve \
  "$MODEL_PATH" \
  --host localhost \
  --port 36124 \
  --backend pytorch \
  --tp_size 1 \
  --ep_size 1 \
  --pp_size 1 \
  --log_level trace \
  --extra_llm_api_options ./gen.yml &> log_gen_36124 &

# Wait for both servers to fully start
sleep 45

# Start the disaggregated coordinator
TRTLLM_VERBOSE=1 trtllm-serve disaggregated -c disagg-config.yml --log_level trace &> log_disagg &

echo "Servers starting... Check logs:"
echo "  Context server: tail -f log_ctx_0"
echo "  Generation server: tail -f log_gen_36124" 
echo "  Disaggregated coordinator: tail -f log_disagg"
echo ""
echo "After startup completes, test with:"
echo "python3 examples/disaggregated/clients/disagg_client.py -c disagg-config.yml -p examples/disaggregated/clients/prompts.json -e completions --max-tokens 10"