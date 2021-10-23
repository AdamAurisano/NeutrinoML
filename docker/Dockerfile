FROM nvidia/cuda:10.2-devel

RUN apt-get update && apt upgrade -y && \
    apt-get install -y wget vim git libopenblas-dev ack htop ncdu net-tools

ENV CONDA_DIR=/usr/local/miniconda3
ENV PATH=$CONDA_DIR/bin:$PATH
ENV TORCH=1.9.0
ENV CUDA_HOME=/usr/local/cuda

RUN wget -nv -O /home/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /home/miniconda.sh -b -p $CONDA_DIR && \
    rm /home/miniconda.sh

RUN conda install -y pytorch=${TORCH}=*cuda10.2* -c pytorch
RUN conda install -y h5py=*=*mpich* -c conda-forge

RUN pip install parameter-sherpa uproot Particle awkward pyarrow \
                tensorboard-data-server boost_histogram seaborn
RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps
RUN conda install -y PyYAML tensorboard psutil jupyterlab numba && \
    conda config --set changeps1 False


