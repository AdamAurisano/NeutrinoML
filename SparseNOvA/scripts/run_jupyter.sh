#!/bin/bash

nohup jupyter lab --port=${USERPORT} > /scratch/SparseNOvA/logs/jupyter.log 2>&1 &

