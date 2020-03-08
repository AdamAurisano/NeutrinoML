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

Once you've pulled down this repository, you can follow instructions for individual projects by reading the instructions inside the corresponding subdirectory of this repository.
