.PHONY: jenkins_quickstart
jenkins_quickstart:
	@$(MAKE) jenkins_run \
		LOCAL_USER=1 \
		DOCKER_RUN_ARGS="-v /home/scratch.trt_llm_data:/home/scratch.trt_llm_data:ro \
-v /home/scratch.rdhar_coreai:/home/scratch.rdhar_coreai:ro" \
		RUN_CMD="bash -lc ' \
echo \"Starting ultra-thorough clean...\" && \
echo \"Cleaning main directories...\" && \
chmod -R u+w build cpp/build cpp/.ccache cpp/.conan 2>/dev/null || true && \
rm -rf build cpp/build cpp/.ccache cpp/.conan 2>/dev/null || true && \
echo \"Force cleaning remaining files...\" && \
find build cpp/build cpp/.ccache cpp/.conan -delete 2>/dev/null || true && \
echo \"Cleaning external dependencies...\" && \
find 3rdparty -name \"build\" -type d -exec rm -rf {} + 2>/dev/null || true && \
find 3rdparty -name \"_deps\" -type d -exec rm -rf {} + 2>/dev/null || true && \
echo \"Cleaning CMake files...\" && \
find . -name \"CMakeCache.txt\" -delete && \
find . -name \"CMakeFiles\" -type d -exec rm -rf {} + 2>/dev/null || true && \
find . -name \"*.d\" -delete 2>/dev/null || true && \
find . -name \"*.o\" -delete 2>/dev/null || true && \
find . -name \"*.so\" -delete 2>/dev/null || true && \
find . -name \"Makefile\" -not -path \"./docker/*\" -delete 2>/dev/null || true && \
echo \"Clean complete, starting build...\" && \
python3 scripts/build_wheel.py \
  --build_type Release \
  --cuda_architectures \"90-real;100-real;120-real\" \
  --install \
  --use_ccache \
  --binding_type nanobind \
  --fast_build \
  --linking_install_binary \
  --skip_building_wheel \
  --clean \
  --clean_wheel && \
pip install -e . && \
exec bash \
'"