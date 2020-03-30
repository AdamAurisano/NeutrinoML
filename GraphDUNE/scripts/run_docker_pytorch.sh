#!/bin/bash

if [ "$1" == "" ]; then
  echo "You must provide an integer argument! Exiting."
  exit
fi

# Get script directory
scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
topdir=`dirname $scriptdir`

# Get docker port prefix based on username
if   [ "$USER" == "hewesje"  ]; then
  export PORTPREFIX=90
elif [ "$USER" == "csarasty" ]; then
  export PORTPREFIX=91
elif [ "$USER" == "haejunoh" ]; then
  export PORTPREFIX=92
elif [ "$USER" == "naporana" ]; then
  export PORTPREFIX=93
else
  echo "Username not recognised! Ask to be added as a user before running Docker."
fi

export USERPORT=${PORTPREFIX}2${1}

nvidia-docker run --name ${USER}-gcn-${1} --expose=${USERPORT} -p ${USERPORT}:${USERPORT} -e USER -e USERPORT -it --rm --shm-size=16g --ulimit memlock=-1 -v ${topdir}:/scratch -v /raid/dune:/data nvcr.io/univcinci/pytorch-gcn:20.03-py3

