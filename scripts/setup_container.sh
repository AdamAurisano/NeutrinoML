#!/bin/bash

conda create -p /opt/neutrinoml python=3.8
source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/neutrinoml
conda install pytorch -c pytorch
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps -v
pip install torch-scatter
pip install torch-sparse
pip install torch-geometric
pip install parameter-sherpa

echo "" >> ~/.bashrc
echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
echo "conda activate /opt/neutrinoml" >> ~/.bashrc
echo "" >> ~/.bashrc

