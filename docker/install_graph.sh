#!/bin/bash

conda install pytorch -c pytorch
export CUDA_HOME=/usr/local/cuda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps
conda install -c conda-forge pytorch_geometric
conda install PyYAML tensorboard
pip install parameter-sherpa

