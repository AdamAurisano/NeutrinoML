#!/bin/bash

if [ "$1" == "" ]; then
  echo "You must provide the path to the summary directory as an argument!"
  exit
fi

nohup tensorboard --logdir $1 --host 0.0.0.0 --port $TENSORBOARD_PORT > logs/tb.log 2>&1 &

