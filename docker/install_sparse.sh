#!/bin/bash

conda install pytorch -c pytorch
export CUDA_HOME=/usr/local/cuda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps
conda install PyYAML tensorboard
pip install parameter-sherpa uproot Particle
conda config --set changeps1 False
cp /scratch/docker/jupyter_notebook_config.py /root/.jupyter

