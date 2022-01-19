#!/usr/bin/env bash

if [ $# != 2 ]; then
  echo "You must provide an integer and an experiment name as argument! Exiting."
  exit
fi

# Get script directory
topdir=/home/$USER/NeutrinoML

# Get docker port prefix based on username
if   [ "$USER" == "hewesje"  ]; then
  export PORT_PREFIX=10
elif [ "$USER" == "csarasty" ]; then
  export PORT_PREFIX=11
elif [ "$USER" == "haejunoh" ]; then
  export PORT_PREFIX=12
elif [ "$USER" == "naporana" ]; then
  export PORT_PREFIX=13
elif [ "$USER" == "rajaoama" ]; then
  export PORT_PREFIX=14
elif [ "$USER" == "byaeggy"  ]; then
  export PORT_PREFIX=15
else
  echo "Username not recognised! Ask to be added as a user before running Docker."
fi

container=jhewes/neutrinoml:pytorch1.10.0-cuda11.3-devel
if [ "$2" == "protodune" ]; then
  datadir=/raid/hewesje/protodune_sparse
  EXPT_ID=0
elif [ "$2" == "nova" ]; then
  datadir=/raid/nova
  EXPT_ID=1
elif [ "$2" == "exatrkx" ]; then
  datadir=/raid
  EXPT_ID=2
elif [ "$2" == "taurnn" ]; then
  datadir=/raid/taurnn
  EXPT_ID=3
else
  echo "Experiment \"${2}\" not recognised! Exiting."
  exit
fi 

export OMP_NUM_THREADS=16

export PORT_PREFIX=${PORT_PREFIX}${EXPT_ID}${1}
export PORT_RANGE=${PORT_PREFIX}0-${PORT_PREFIX}9
export JUPYTER_PORT=${PORT_PREFIX}0
export SHERPA_PORT=${PORT_PREFIX}1
export TENSORBOARD_PORT=${PORT_PREFIX}2

nvidia-docker run --name ${USER}-${2}-${1} --expose=${PORT_RANGE} -p ${PORT_RANGE}:${PORT_RANGE} -e USER -e OMP_NUM_THREADS -e PORT_PREFIX -e JUPYTER_PORT -e SHERPA_PORT -e TENSORBOARD_PORT -e PYTHONPATH=/scratch:/numl:/usr/local/pynuml:/usr/local/pandana:/usr/local/NOvAPandAna -it --rm --shm-size=16g --ulimit memlock=-1 -v ${topdir}:/scratch -v ${datadir}:/data -v /home/${USER}/pynuml:/numl --workdir /scratch $container

