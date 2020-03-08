#!/bin/bash

if [ "$1" == "" ]; then
  echo "You must provide an integer argument! Exiting."
  exit
fi

# Get script directory
scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
topdir=`dirname $scriptdir`

nvidia-docker run --name ${USER}-scn-${1} --expose=900${1} -p 900${1}:900${1} -e USER -it --rm --shm-size=16g --ulimit memlock=-1 -v ${topdir}:/scratch -v /raid/hewesje/protodune_sparse:/data nvcr.io/univcinci/pytorch-sparseconv:19.12-py3

