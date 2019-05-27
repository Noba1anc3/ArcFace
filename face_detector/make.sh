#!/usr/bin/env bash
cd ./utils/

CUDA_PATH=/usr/local/cuda/

python3.6 build.py build_ext --inplace

cd ..
