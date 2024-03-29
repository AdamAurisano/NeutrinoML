# Creating a new docker image

Our docker image is based in PyTorch, and uses a variety of custom packages - MinkowskiEngine for sparse convolutions, PyTorch Geometric for graph convolutions, etc. This page contains instructions on how to produce a container image from scratch, and commit it for others to use.

## Image from Dockerfile

Building from the Dockerfile with `nvidia-docker build .` will compile the image in place. Due to issues with GPU access during container building, MinkowskiEngine must be compiled interactively in the image _after_ compilation. You can do this by running `./run -i <image id>` to start a container based on the image that was just built, and then `source minkowski.sh` inside the container to install MinkowskiEngine. You can then detach from the container and commit it as a new image.

## Manual image

Running `bash new_image.sh` will spin up a new container based on an image containing CUDA 10.2. From there, you can install some important packages and then install Anaconda inside the image by running `bash setup_conda.sh`, answering "yes" when prompted. The new conda environment will not be enabled by default, so you should then run `exec bash` to refresh the terminal. You can proceed to install PyTorch and other custom packages by running `bash install_packages.sh`. Once complete, you can then detach from the image and commit it by running

```
docker commit <image id> nvcr.io/univcinci/pytorch-neutrinoml:<tag>
```

where the image ID can be found by running `docker ps`, while the desired tag can be whatever the user chooses. The current default is `1.8`, but note that committing a new image with an already used tag will overwrite the existing image, so do so with caution.

