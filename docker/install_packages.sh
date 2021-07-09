#!/bin/bash
rm /home/Miniconda3-latest-Linux-x86_64.sh
conda install -y pytorch=1.9.0 -c pytorch
export CUDA_HOME=/usr/local/cuda
pip install parameter-sherpa uproot Particle awkward pyarrow tensorboard-data-server boost_histogram seaborn
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps
conda install -y PyYAML tensorboard h5py mpi4py psutil jupyterlab numba
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
export PYTHONPATH=/scratch:/usr/local/pandana:/usr/local/NOvAPandAna:/usr/local/pynuml
cd /usr/local
git clone https://github.com/jethewes/pynuml
git clone https://github.com/HEPonHPC/pandana
git clone https://github.com/grohmc/NOvAPandAna

