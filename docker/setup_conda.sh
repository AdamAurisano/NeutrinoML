#!/bin/bash

apt update
apt upgrade -y
apt install -y wget vim git libopenblas-dev
cd /home
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh

