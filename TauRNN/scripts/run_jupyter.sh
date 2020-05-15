#!/bin/bash

nohup jupyter lab --port=${USERPORT} > /scratch/TauRNN/logs/jupyter.log 2>&1 &

