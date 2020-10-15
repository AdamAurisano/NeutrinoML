#!/bin/bash
nohup jupyter lab --port=${JUPYTER_PORT} > /scratch/logs/jupyter.log 2>&1 &

