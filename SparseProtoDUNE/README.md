# ProtoDUNE sparse segmentation

These instructions assume you're working on Heimdall, and have access to the relevant inputs. In order to start, you must run

```
bash scripts/run_docker_pytorch.sh <N>
```

where `<N>` is the last digit of the port number you wish to use. If you aren't sure what this means, you should reach out to me (Jeremy) to coordinate which ports to use. Currently reserved ports are:

90XX – Jeremy
91XX – Carlos
92XX – Stella
93XX – Nicole

This script will launch a Docker container containing the environment necessary to train, mounting the correct data directory on Heimdall and this directory (SparseProtoDUNE) as your working directory.

Inside your Docker container, you should navigate to your `/scratch` working directory and then copy down the SparseConvNet framework by running

```
git clone https://github.com/facebookresearch/SparseConvNet
```

Once this is done, you should run

```
source scripts/source_me.sh
```

to set up your working environment and compile `SparseConvNet` (this will take a few minutes).

You should run 

```
python scripts/train.py
```

to train a network, and

```
scripts/inference.py
```

to make plots to benchmark the network.

## Monitoring training

You can also launch a tensorboard instance by running

```
bash scripts/run_tb.sh
```

although you will need to modify the port number in this script to match the values you chose when launching your Docker container. You can modify the train and test configuration by editing `config/sparse_standard.yaml` or creating alternate config files to pass to the training and inference scripts.

