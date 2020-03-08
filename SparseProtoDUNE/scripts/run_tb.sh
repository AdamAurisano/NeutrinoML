#!/bin/bash

nohup tensorboard --logdir $PWD/summary --port ${USERPORT} > logs/tb.log 2>&1 &

