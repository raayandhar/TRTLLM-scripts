build_tensorrt_llm () {
    tensorrt_llm_dir=$1
    tensorrt_llm_commit=${2:-HEAD}
    cd $tensorrt_llm_dir
    git lfs update --force
    git lfs install && git lfs pull
    git checkout -- .
    git checkout $tensorrt_llm_commit
    apt-get --reinstall install libibverbs-dev
    rm -rf cpp/build/*
    rm -rf build/
    rm -rf .venv-3.12/
    rm -rf $tensorrt_llm_dir/build/tensorrt_llm-*.whl
    python3 ./scripts/build_wheel.py --configure_cmake --use_ccache -b Release -a 100-real --benchmarks --trt_root /usr/local/tensorrt --extra-cmake-vars NVTX_DISABLE=OFF --extra-cmake-vars ENABLE_MULTI_DEVICE=1 -i
    pip3 install $tensorrt_llm_dir/build/tensorrt_llm-*.whl
}

tensorrt_llm_dir=${1:-$(pwd)}
tensorrt_llm_commit=${2:-HEAD}
path_to_current_script=$(realpath $0)

if [ -z $SLURM_JOB_ID ]; then
    container_image=urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-25.05-py3-x86_64-ubuntu24.04-trt10.11.0.33-skip-tritondevel-202507162011-ec3ebae
    if [ -z $container_image ]; then
        echo "Error: Could not find container image in Jenkins configuration"
        exit 1
    fi
    echo $container_image
    tensorrt_llm_commit=$(git rev-parse HEAD)
    tensorrt_llm_commit=$(echo $tensorrt_llm_commit | cut -c1-7)

    srun -A coreai_comparch_infbench --time=00:20:00 --container-mounts $tensorrt_llm_dir:$tensorrt_llm_dir -N1 -p batch -J coreai_comparch_infbench-trtllm-update-image:update-image --container-image=$container_image --container-save=$tensorrt_llm_dir/$tensorrt_llm_commit.sqsh bash -ex -c "bash $path_to_current_script $tensorrt_llm_dir $tensorrt_llm_commit"
else
    build_tensorrt_llm $tensorrt_llm_dir $tensorrt_llm_commit
fi