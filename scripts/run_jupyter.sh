#!/bin/bash
nohup jupyter lab --port=${JUPYTER_PORT} > /scratch/logs/jupyter_${JUPYTER_PORT}.log 2>&1 &

