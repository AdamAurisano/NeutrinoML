# NeutrinoML
Repository for Aurisano Group Machine Learning Projects

## Accessing Heimdall
If you don't have the ability to log into Heimdall, contact Adam to be given accounts.

Once you have accounts, you can log into Heimdall by setting up your SSH configuration for easy access. To do this, add the following to `~/.ssh.config`:

```
Host earth
  HostName earth.phy.uc.edu
  User <your username>

Host hd
  HostName heimdall.geop.uc.edu
  User <your username>
  ProxyJump earth
```

You should then be able to log into Heimdall by doing

```
ssh hd
```

When you first log in, you should be in your own `/home/<username>` directory. From there, you can clone this repository:

```
git clone https://github.com/AdamAurisano/NeutrinoML
```

I'm assuming you already have access if you're reading this, but if you don't, you can ask Adam to give you access.

## Setting up ports

In computer networking, a *port* is a communication endpoint on a machine. A procedure called *port forwarding* allows you to connect up ports on your local machine to ports on Heimdall. Port forwarding will allow you to run web pages on Heimdall, and then access them easily on your local machine. For instance, you can run the program *tensorboard*, which provides web pages used to monitor training, and then access and monitor that webpage on your local machine by forwarding ports.

Part of getting set up in this workflow is being assigned ports. Since ports can collide if used by multiple people or applications, each user is assigned their own set of ports. The bookkeeping method for port usage is sociological rather than technological – each user is given a range of numbers representing ports that "belong to them", and as long as they use only those numbers, no collisions should occur. Ports are five-digit numbers, and the user will be assigned a unique pair of starting digits. All 100 ports beginning with this prefix belong to the user.

When you start a new Docker container, a set of ports will be automatically assigned. The five-digit port number is defines as

```
AABCD
```

where `AA` is a unique pair of digits associated with each user:
* 10 - jhewes15
* 11 - csarasty
* 12 - haujunoh
* 13 - naporana
* 14 - rajaoama
* 15 - byaeggy

`B` is an integer associated with each workflow:
* 0 - protodune
* 1 - nova
* 2 - dunegraph
* 3 - taurnn

`C` is the integer you passed to `run_docker_pytorch.sh`, and the final integer `D` refers to a specific application within the docker container. All possible values of integer `D` are exposed and forwarded from Docker (ie. AABC0-AABC9), but only some are assigned a specific purpose. Inside your container, the environment variables `JUPYTER_PORT`, `SHERPA_PORT` and `TENSORBOARD_PORT` are automatically set, and your code will automatically use these ports. To query a specific port, you can do 

```
echo $JUPYTER_PORT
```

In order to access these ports locally, they will need to be forwarded in your local SSH configuration. You can set up port forwarding from your local machine by adding lines to your .ssh config as follows:

```
Host hd
  HostName heimdall.geop.uc.edu
  User <your username>
  ProxyJump earth
  LocalForward 1234 localhost:1234
```
where `1234` should here be replaced with whichever port number you wish to forward. In the case you wish to forward multiple ports at the same time, you can add as many `LocalForward` lines in succession as you wish with no restrictions.

Once you've pulled down this repository, you can follow instructions for individual projects by reading the instructions inside the corresponding subdirectory of this repository.
