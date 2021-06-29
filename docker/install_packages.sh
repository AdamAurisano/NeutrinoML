#!/bin/bash

conda install -y pytorch=1.9.0 -c pytorch
export CUDA_HOME=/usr/local/cuda
pip install parameter-sherpa uproot Particle awkward pyarrow tensorboard-data-server boost_histogram
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps
conda install -y PyYAML tensorboard h5py mpi4py psutil
conda config --set changeps1 False
TORCH=torch-1.9.0
CUDA=cu102
pip install torch-scatter -f https://pytorch-geometric.com/whl/${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/${TORCH}+${CUDA}.html
pip install torch-geometric
mkdir /root/.jupyter
cp /scratch/jupyter_notebook_config.py /root/.jupyter
export OMP_NUM_THREADS=16
cd /home
git clone https://github.com/HEPonHPC/pandana
export PYTHONPATH=/home/pandana

