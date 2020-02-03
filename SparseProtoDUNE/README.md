# ProtoDUNE sparse segmentation

These instructions assume you're working on Heimdall, and have access to the relevant inputs. In order to start, you must run

```
bash scripts/run_docker_pytorch.sh
```

This script will launch a docker container containing the environment necessary to train, mounting the correct data directory on Heimdall and this directory (SparseProtoDUNE) as your working directory. Before running, you should modify the script to change the default 900X ports to your own ones - speak to me (Jeremy) about how to do this.

You should run `python scripts/train.py` to train a network, and `scripts/inference.py` to make plots to benchmark the network. You can also launch a tensorboard instance by running `bash scripts/run_tb.sh`, although you will need to modify the port number in this script to match the values you chose when launching your docker container. You can modify the train and test configuration by editing `config/sparse_standard.yaml` or creating alternate config files to pass to the training and inference scripts.

