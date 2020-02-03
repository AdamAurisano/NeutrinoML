#!/bin/bash

nohup tensorboard --logdir $PWD/summary --port 9002 > logs/tb.log 2>&1 &

