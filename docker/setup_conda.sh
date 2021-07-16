#!/bin/bash

apt update
apt upgrade -y
apt install -y wget vim git libopenblas-dev ack htop ncdu net-tools
wget -P /home https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /home/Miniconda3-latest-Linux-x86_64.sh

