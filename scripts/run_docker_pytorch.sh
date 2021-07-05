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

container=nvcr.io/univcinci/pytorch-neutrinoml:1.9-dev
if [ "$2" == "protodune" ]; then
  datadir=/raid/hewesje/protodune_sparse
  exptport=0
elif [ "$2" == "nova" ]; then
  datadir=/raid/nova
  exptport=1
elif [ "$2" == "dune" ]; then
  datadir=/raid
  exptport=2
elif [ "$2" == "taurnn" ]; then
  datadir=/raid/taurnn
  exptport=3
else
  echo "Experiment \"${2}\" not recognised! Exiting."
  exit
fi 

export OMP_NUM_THREADS=16

export PORT=${PORTPREFIX}${exptport}${1}
export PORT_RANGE=${PORT}0-${PORT}9
export JUPYTER_PORT=${PORT}0
export SHERPA_PORT=${PORT}1
export TENSORBOARD_PORT=${PORT}2

nvidia-docker run --name ${USER}-${2}-${1} --expose=${PORT_RANGE} -p ${PORT_RANGE}:${PORT_RANGE} -e USER -e OMP_NUM_THREADS -e JUPYTER_PORT -e SHERPA_PORT -e TENSORBOARD_PORT -e PYTHONPATH=/scratch:/usr/local/pandana:/usr/local/NOvAPandAna -it --rm --shm-size=16g --ulimit memlock=-1 -v ${topdir}:/scratch -v ${datadir}:/data -v /home/${USER}/pynuml:/numl --workdir /scratch $container

