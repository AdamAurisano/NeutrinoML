#!/bin/bash

conda install pytorch -c pytorch
export CUDA_HOME=/usr/local/cuda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps
pip install PyYAML parameter-sherpa torch-scatter torch-sparse torch-geometric

