#!/bin/bash
nvidia-docker run --name ${USER}-pandana -e USER -e OMP_NUM_THREADS -it --rm --shm-size=16g --ulimit memlock=-1 -v /raid/nova:/data --workdir /home nvcr.io/univcinci/pandana:latest
