#!/bin/bash

if [ "$1" == "" ]; then
  echo "You must provide an integer argument! Exiting."
  exit
fi

# Get script directory
topdir=/home/$USER/NeutrinoML # is this hack going to work?

# Get docker port prefix based on username
if   [ "$USER" == "hewesje"  ]; then
  export PORTPREFIX=90
elif [ "$USER" == "csarasty" ]; then
  export PORTPREFIX=91
elif [ "$USER" == "haejunoh" ]; then
  export PORTPREFIX=92
elif [ "$USER" == "naporana" ]; then
  export PORTPREFIX=93
elif [ "$USER" == "rajaoama" ]; then
  export PORTPREFIX=94
else
  echo "Username not recognised! Ask to be added as a user before running Docker."
fi

export USERPORT=${PORTPREFIX}3${1}

nvidia-docker run --name ${USER}-tau-${1} --expose=${USERPORT} -p ${USERPORT}:${USERPORT} -e USER -e USERPORT -it --rm --shm-size=16g --ulimit memlock=-1 -v ${topdir}:/scratch -v /raid/taurnn:/data nvcr.io/univcinci/pytorch-sparseconv:20.03-py3
