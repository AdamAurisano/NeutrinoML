#!/usr/bin/env bash

if [ $# != 2 ]; then
  echo "You must provide an integer and an experiment name as argument! Exiting."
  exit
fi

# Get script directory
topdir=/home/$USER/NeutrinoML

# Get docker port prefix based on username
if   [ "x$USER" == "xhewesje"  ]; then
  export PORTPREFIX=10
elif [ "x$USER" == "xcsarasty" ]; then
  export PORTPREFIX=11
elif [ "x$USER" == "xhaejunoh" ]; then
  export PORTPREFIX=12
elif [ "x$USER" == "xnaporana" ]; then
  export PORTPREFIX=13
elif [ "x$USER" == "xrajaoama" ]; then
  export PORTPREFIX=14
elif [ "x$USER" == "xbyaeggy"  ]; then
  export PORTPREFIX=15
elif [ "x$USER" == "xdvargas" ]; then
  export PORTPREFIX=16
 elif [ "x$USER" == "xwshi" ]; then
  export PORTPREFIX=17
else
  echo "Username not recognised! Ask to be added as a user before running Docker."
fi

container=jhewes/neutrinoml:pytorch1.10.0-cuda11.3-devel
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

nvidia-docker run --name ${USER}-${2}-${1} --expose=${PORT_RANGE} -p ${PORT_RANGE}:${PORT_RANGE} -e USER -e OMP_NUM_THREADS -e JUPYTER_PORT -e SHERPA_PORT -e TENSORBOARD_PORT -e PYTHONPATH=/scratch:/numl:/usr/local/pynuml:/usr/local/pandana:/usr/local/NOvAPandAna -it --rm --shm-size=16g --ulimit memlock=-1 -v ${topdir}:/scratch -v ${datadir}:/data -v /home/${USER}/pynuml:/numl --workdir /scratch $container

