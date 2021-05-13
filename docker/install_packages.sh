#!/bin/bash

conda install pytorch=1.8.1 -c pytorch
export CUDA_HOME=/usr/local/cuda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps
conda install PyYAML tensorboard
pip install parameter-sherpa uproot Particle awkward pyarrow
conda config --set changeps1 False
TORCH=torch-1.8.1
CUDA=cu102
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
mkdir /root/.jupyter
cp /scratch/jupyter_notebook_config.py /root/.jupyter
export OMP_NUM_THREADS=16

