.PHONY: jenkins_quickstart
jenkins_quickstart:
	@$(MAKE) jenkins_run \
		LOCAL_USER=1 \
		DOCKER_RUN_ARGS="-v /home/scratch.trt_llm_data:/home/scratch.trt_llm_data:ro \
-v /home/scratch.rdhar_coreai:/home/scratch.rdhar_coreai:ro" \
		RUN_CMD="bash -lc ' \
python3 scripts/build_wheel.py \
  --build_type Release \
  --cuda_architectures \"90-real;100-real\" \
  --install \
  --use_ccache \
  --fast_build \
  --linking_install_binary \
  --skip_building_wheel && \
pip install -e . && \
exec bash \
'"