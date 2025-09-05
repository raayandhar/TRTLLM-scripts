Some helpful things from the slack DMs:

```
mpirun -n 2 trtllm-serve disaggregated_mpi_worker -c examples/disaggregated/disagg_config.yaml > mpi_servers.log 2>&1 &
trtllm-serve disaggregated -c examples/disaggregated/disagg_config.yaml > mpi_servers_disagg.log 2>&1 &
```

```
export TRTLLM_USE_MPI_KVCACHE=1
export PATH=$HOME/.local/bin:$PATH
export TLLM_LOG_LEVEL=debug
```

```
python3 examples/disaggregated/clients/disagg_client.py -c examples/disaggregated/disagg_config.yaml -p examples/disaggregated/clients/prompts.json -e completions --max-tokens 10
```

```
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LLM_MODELS_ROOT=/home/scratch.trt_llm_data/llm-models/
export PATH=$HOME/.local/bin:$PATH
export TLLM_LOG_LEVEL=debug
```

`export TRTLLM_ACCURACY_NO_REFERENCE=1`

```
CUDA_VISIBLE_DEVICES=0 trtllm-serve /home/scratch.trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct --host localhost --port 8001 --extra_llm_api_options ./ctx.yaml &> log_ctx_0 &
CUDA_VISIBLE_DEVICES=1 trtllm-serve /home/scratch.trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct --host localhost --port 8002 --extra_llm_api_options ./gen.yaml &> log_gen_0 &
trtllm-serve disaggregated -c disagg_config.yaml &> log_disagg & 
```

`python3 examples/disaggregated/clients/disagg_client.py -c disagg_config.yaml -p examples/disaggregated/clients/prompts.json -e completions --max-tokens 10`

```
cd cpp/build && cmake .. -DNIXL_ROOT=/opt/nvidia/nvda_nixl -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) transferAgentTest

make tensorrt_llm_nixl_wrapper -j$(nproc) && export LD_LIBRARY_PATH=$(pwd)/tensorrt_llm/executor/cache_transmission/nixl_utils:$LD_LIBRARY_PATH && ./tests/unit_tests/executor/transferAgentTest &&
export NIXL_ROOT=/opt/nvidia/nvda_nixl

# Set the plugin directory explicitly
export NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu/plugins

# Also ensure the base library path is set
export LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu:/opt/nvidia/nvda_nixl/lib64:$LD_LIBRARY_PATH

# Test again
./nixl_gds_test /tmp/nixl_test
```

`squeue -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R" -w viking-prod-323`
