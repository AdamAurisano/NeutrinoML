#!/bin/bash

nohup jupyter lab --port=${USERPORT} > logs/jupyter.log 2>&1 &

