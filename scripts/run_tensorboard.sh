#!/bin/bash

if [ "$1" == "" ]; then
  echo "You must provide the path to the summary directory as an argument!"
  exit
fi

nohup tensorboard --logdir $1 --port $TENSORBOARD_PORT > /dev/null 2>&1 &

