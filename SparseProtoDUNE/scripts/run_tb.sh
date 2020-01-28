#!/bin/bash

nohup tensorboard --logdir $PWD/summary --port 9001 > logs/tb.log &

