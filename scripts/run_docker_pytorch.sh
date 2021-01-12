#!/bin/bash

if [ "$2" == "" ]; then
  echo "You must provide an integer and an experiment name as argument! Exiting."
  exit
fi

# Get script directory
topdir=/home/$USER/NeutrinoML

# Get docker port prefix based on username
if   [ "$USER" == "hewesje"  ]; then
  export PORTPREFIX=10
elif [ "$USER" == "csarasty" ]; then
  export PORTPREFIX=11
elif [ "$USER" == "haejunoh" ]; then
  export PORTPREFIX=12
elif [ "$USER" == "naporana" ]; then
  export PORTPREFIX=13
elif [ "$USER" == "rajaoama" ]; then
  export PORTPREFIX=14
elif [ "$USER" == "byaeggy"  ]; then
  export PORTPREFIX=15
else
  echo "Username not recognised! Ask to be added as a user before running Docker."
fi

container=nvcr.io/univcinci/pytorch-neutrinoml:20.12-py3
if [ "$2" == "protodune" ]; then
  datadir=/raid/hewesje/protodune_sparse
  exptport=0
elif [ "$2" == "nova" ]; then
  datadir=/raid/nova
  exptport=1
elif [ "$2" == "dunegraph" ]; then
  datadir=/raid/dune
  exptport=2
elif [ "$2" == "taurnn" ]; then
  datadir=/raid/taurnn
  exptport=3
else
  echo "Experiment \"${2}\" not recognised! Exiting."
  exit
fi 

export JUPYTER_PORT=${PORTPREFIX}${exptport}${1}0
export SHERPA_PORT=${PORTPREFIX}${exptport}${1}1
export TENSORBOARD_PORT=${PORTPREFIX}${exptport}${1}2

nvidia-docker run --name ${USER}-${2}-${1} --expose=${JUPYTER_PORT} -p ${JUPYTER_PORT}:${JUPYTER_PORT} --expose=${SHERPA_PORT} -p ${SHERPA_PORT}:${SHERPA_PORT} --expose=${TENSORBOARD_PORT} -p ${TENSORBOARD_PORT}:${TENSORBOARD_PORT} -e USER -e JUPYTER_PORT -e SHERPA_PORT -it --rm --shm-size=16g --ulimit memlock=-1 -v ${topdir}:/scratch -v ${datadir}:/data --workdir /scratch $container

