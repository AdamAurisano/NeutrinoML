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
else
  echo "Username not recognised! Ask to be added as a user before running Docker."
fi

if [ "$2" == "protodune" ]; then
  datadir=/raid/hewesje/protodune_sparse
  container=nvcr.io/univcinci/pytorch-sparseconv:19.12-py3
  exptport=0
elif [ "$2" == "nova" ]; then
  datadir=/raid/nova
  container=nvcr.io/univcinci/pytorch-neutrinoml:20.03-py3
  exptport=1
elif [ "$2" == "dunegraph" ]; then
  datadir=/raid/dune
  container=nvcr.io/univcinci/pytorch-gcn:20.03-py3
  exptport=2
elif [ "$2" == "taurnn" ]; then
  datadir=/raid/taurnn
  container=nvcr.io/univcinci/pytorch-neutrinoml:20.03-py3
  exptport=3
else
  echo "Experiment \"${2}\" not recognised! Exiting."
  exit
fi 

export JUPYTER_PORT=${PORTPREFIX}${exptport}${1}0
export SHERPA_PORT=${PORTPREFIX}${exptport}${1}1

nvidia-docker run --name ${USER}-${2}-${1} --expose=${JUPYTER_PORT} -p ${JUPYTER_PORT}:${JUPYTER_PORT} --expose=${SHERPA_PORT} -p ${SHERPA_PORT}:${SHERPA_PORT} -e USER -e JUPYTER_PORT -e SHERPA_PORT -it --rm --shm-size=16g --ulimit memlock=-1 -v ${topdir}:/scratch -v ${datadir}:/data $container

