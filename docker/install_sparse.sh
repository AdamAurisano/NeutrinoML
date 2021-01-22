#!/bin/bash

conda install pytorch -c pytorch
export CUDA_HOME=/usr/local/cuda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps
conda install PyYAML tensorboard
pip install parameter-sherpa uproot Particle
conda config --set changeps1 False
mkdir /root/.jupyter
cp /scratch/jupyter_notebook_config.py /root/.jupyter
export OMP_NUM_THREADS=16

