FROM nvidia/cuda:11.3.1-devel

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y wget vim git ack htop ncdu net-tools libopenblas-dev

ENV CONDA_DIR=/usr/local/miniconda3
ENV PATH=$CONDA_DIR/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH=1.10.0
ENV OMP_NUM_THREADS=16

RUN wget -nv -O /home/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /home/miniconda.sh -b -p $CONDA_DIR && rm /home/miniconda.sh

RUN conda install -y pytorch=$TORCH=*cuda11.3* torchvision cudatoolkit h5py=*=*mpich* -c pytorch -c conda-forge

RUN pip install parameter-sherpa uproot Particle awkward pyarrow \
                tensorboard-data-server boost_histogram seaborn
RUN conda install -y PyYAML tensorboard psutil jupyterlab numba plotly && \
    conda config --set changeps1 False

ARG GEO_CUDA=cu113
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${GEO_CUDA}.html && \
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${GEO_CUDA}.html && \
    pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${GEO_CUDA}.html && \
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${GEO_CUDA}.html && \
    pip install torch-geometric

RUN mkdir /root/.jupyter
COPY jupyter_notebook_config.py /root/.jupyter

RUN cd /usr/local && git clone https://github.com/jethewes/pynuml && \
    git clone https://github.com/HEPonHPC/pandana
ENV PYTHONPATH=/scratch:/usr/local/pandana:/usr/local/pynuml

