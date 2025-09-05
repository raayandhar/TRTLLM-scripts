#!/usr/bin/env bash
BASE_DIR="/home/scratch.rdhar_coreai/TRTLLM"
function show_error {
  echo "Has to be a valid option under"
  ls "$BASE_DIR"
}
if [ -z "$1" ]; then
  show_error
  exit 1
fi
TARGET_DIR="$BASE_DIR/$1/TensorRT-LLM"
if [ ! -d "$TARGET_DIR" ]; then
  show_error
  exit 1
fi
cd "$TARGET_DIR"
make -C docker jenkins_quickstart HOME_DIR="/home/scratch.rdhar_coreai"
#cd /home/scratch.rdhar_coreai/disagg_pp/TensorRT-LLM
#make -C docker jenkins_quickstart HOME_DIR=/home/scratch.rdhar_coreai
