# ProtoDUNE sparse segmentation

These instructions assume you're working on Heimdall, and have access to the relevant inputs. In order to start, you must run

```
scripts/run_docker_pytorch.sh N
```

where `N` should be replaced with the last digit of the port number you wish to use. If you aren't sure what this means, you should reach out to me (Jeremy) to coordinate which ports to use. Currently reserved ports are:

90XX – Jeremy

91XX – Carlos

92XX – Stella

93XX – Nicole

This script will launch a Docker container containing the environment necessary to train, mounting the correct data directory on Heimdall and this directory (SparseProtoDUNE) as your working directory.

Inside your Docker container, you should navigate to your working directory by doing

```
cd /scratch
```

Once this is done, you should run

```
source scripts/source_me.sh
```

to set up your working environment. Once that's done, you should be good to start training.

## Running training

The Heimdall machine has eight GPU cards, each of which has 16GB of memory available. Most ML tasks are configured to fully utilise the memory of whatever GPUs they run on, so if somebody is already making use of a specific card, you can think of it as "already in use" and use another one instead. If you try to spin up training on a GPU somebody else is using, that usually leads to your training failing due to an out-of-memory error.

You can run the command `nvidia-smi` to check which GPUs are already in use, and choose an unoccupied one to train.

Configuration files for running are stored in the `config` directory. By default, if no configuration is specified, the config `config/sparse_standard.yaml` will be used by default. The GPU can be changed by changing the `device` parameter in the configuration, and should be a number between `0` and `7`. These numbers correspond to those listed by the `nvidia-smi` command.

You can set up training by running the command

```
scripts/train.py
```

and run inference by running

```
scripts/inference.py
```

to make plots to benchmark the network.

You can modify the train and test configuration by editing `config/sparse_standard.yaml` or creating alternate config files to pass to the training and inference scripts.

## Monitoring training

You can launch a tensorboard instance by running

```
scripts/run_tb.sh
```

which will be exposed on the port automatically configured when the Docker container was started. You can then navigate to `localhost:XXXX`, where `XXXX` is your port number, in a web browser on your local machine to monitor training.
