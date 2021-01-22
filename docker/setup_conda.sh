#!/bin/bash

apt update
apt upgrade -y
apt install -y wget vim git libopenblas-dev
wget -P /home https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash /home/Anaconda3-2020.11-Linux-x86_64.sh

