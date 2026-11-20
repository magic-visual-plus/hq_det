#!/bin/bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export FORCE_CUDA=1
export MMCV_WITH_OPS=1
export TORCH_CUDA_ARCH_LIST="8.9"
/home/envs/phi/bin/pip install --no-build-isolation mmcv==2.1.0 > /home/hq_det/mmcv_build.log 2>&1
echo "EXIT=$?" >> /home/hq_det/mmcv_build.log
