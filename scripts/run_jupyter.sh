#!/bin/bash
nohup jupyter lab --port=$JUPYTER_PORT > /dev/null 2>&1 &

