#!/bin/bash

BATCH_SIZE=8
EAGLE=false

if [ $# -eq 0 ]; then
    echo "Usage: $0 <partition> [<eagle>]"
    echo "Example: $0 batch true"
    exit 1
fi

PARTITION=$1

if [ $# -eq 2 ]; then
    EAGLE=$2
fi

ACCOUNT="coreai_comparch_infbench"
CONTAINER_IMAGE="/lustre/fsw/coreai_comparch_infbench/rdhar/ae28b3a.sqsh"
MAVERICK_PATH="/lustre/fsw/coreai_comparch_infbench/rdhar/weights/maverick"
EAGLE_PATH="/lustre/fsw/coreai_comparch_infbench/rdhar/weights/eagle"
RUN_DIR="/lustre/fsw/coreai_comparch_infbench/rdhar/llama4/new_runs/benchmark_$(date +%Y%m%d-%H%M%S)"

TP_SIZE=8
EP_SIZE=1
NUM_REQUESTS=32
INPUT_SEQ_LENGTH=1024
OUTPUT_SEQ_LENGTH=2048
DATASET_PATH="/lustre/fsw/coreai_comparch_infbench/rdhar/aa_gen_16k_1k_apr30_no_system_prompt.txt"
PROFILE_RANGE="800-810"

mkdir -p ${RUN_DIR}/logs
cd ${RUN_DIR}

if [ "$EAGLE" = true ]; then
    cat > extra-llm-api-config.yaml << 'EOF'
disable_overlap_scheduler: false
enable_autotuner: false
enable_attention_dp: false
enable_min_latency: true
cuda_graph_config:
  max_batch_size: 8
speculative_config:
  decoding_type: Eagle
  max_draft_len: 3
  speculative_model_dir: /config/models/eagle
  eagle3_one_model: true
kv_cache_config:
  enable_block_reuse: false
EOF
    echo "MODEL_DIR: ${MAVERICK_PATH}" > model_dir.txt
    echo "DRAFT_MODEL_DIR: ${EAGLE_PATH}" >> model_dir.txt
else
    cat > extra-llm-api-config.yaml << 'EOF'
disable_overlap_scheduler: false
enable_autotuner: false
enable_attention_dp: false
enable_min_latency: true
cuda_graph_config:
  max_batch_size: 8
kv_cache_config:
  enable_block_reuse: false
EOF
    echo "MODEL_DIR: ${MAVERICK_PATH}" > model_dir.txt
    echo "DRAFT_MODEL_DIR: None" >> model_dir.txt
fi

echo "DATASET: ${DATASET_PATH}" > dataset.txt
echo "HOSTNAME: $(hostname)" > machine.txt
echo "TIMESTAMP: $(date +%Y%m%d-%H%M%S)" >> machine.txt

cat > launch_benchmark.sbatch << EOF
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/benchmark.out
#SBATCH -e logs/benchmark.err
#SBATCH -J coreai_comparch_infbench-benchmark:llama4_eagle3

echo "Starting TensorRT-LLM benchmark..."
echo "Capturing profiles."
echo "Eagle enabled: ${EAGLE}"
echo "Concurrency = Batch size = ${BATCH_SIZE}"
echo "TP/EP: ${TP_SIZE}/${EP_SIZE}"
echo "Model directory: /config/models/maverick"
echo "Draft model directory: /config/models/eagle"
echo "Dataset: ${DATASET_PATH}"
echo "Profile range: ${PROFILE_RANGE}"
echo "########################################################"

export NOPE_LAYER_INTERVAL=4
export TRTLLM_ENABLE_PDL=1
export TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=True
export TLLM_PROFILE_START_STOP=${PROFILE_RANGE}

srun -l \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=${MAVERICK_PATH}:/config/models/maverick,${EAGLE_PATH}:/config/models/eagle,${RUN_DIR}:/workspace,${DATASET_PATH}:/dataset.txt \
    --container-workdir=/workspace \
    --export=ALL,NOPE_LAYER_INTERVAL=4,TRTLLM_ENABLE_PDL=1,TLLM_PROFILE_START_STOP=${PROFILE_RANGE} \
    --mpi=pmix \
    bash -c "
        set -ex
        export PATH=\$PATH:~/.local/bin

        echo 'Nsight Systems version:'
        nsys --version

        nsys profile \
            --output=/workspace/logs/profile_\$(hostname)_\$(date +%Y%m%d_%H%M%S)_BS_${BATCH_SIZE}_eagle_${EAGLE}.nsys-rep \
            --force-overwrite=true \
            --trace=cuda,nvtx,osrt \
            --cuda-memory-usage=true \
            --cuda-graph-trace=node \
            -c cudaProfilerApi --capture-range-end=repeat[] \
            trtllm-llmapi-launch trtllm-bench \
            -m /config/models/maverick \
            --model_path /config/models/maverick \
            throughput \
            --backend pytorch \
            --max_batch_size ${BATCH_SIZE} \
            --max_num_tokens $((${INPUT_SEQ_LENGTH} + ${OUTPUT_SEQ_LENGTH})) \
            --extra_llm_api_options /workspace/extra-llm-api-config.yaml \
            --dataset /dataset.txt \
            --warmup 1 \
            --streaming \
            --num_requests ${NUM_REQUESTS} \
            --concurrency ${BATCH_SIZE} \
            --tp ${TP_SIZE} \
            --ep ${EP_SIZE} \
            --eos_id 200008 \
            2>&1 | tee logs/trtllmbench_latency_TP_${TP_SIZE}_EP_${EP_SIZE}_BS_${BATCH_SIZE}_ISL_${INPUT_SEQ_LENGTH}_OSL_${OUTPUT_SEQ_LENGTH}_\$(hostname)_\$(date +%Y%m%d_%H%M%S)_eagle_${EAGLE}.log
    "
EOF

echo "Submitting benchmark job..."
echo "Results will be in: ${RUN_DIR}"
sbatch launch_benchmark.sbatch
