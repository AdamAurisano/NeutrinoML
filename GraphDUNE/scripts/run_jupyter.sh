#!/bin/bash
nohup jupyter lab --port=${USERPORT} > /scratch/GraphDUNE/logs/jupyter.log 2>&1 &

