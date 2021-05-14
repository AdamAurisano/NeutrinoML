nvidia-docker run --name ${USER}-dev -it --rm --shm-size=16g --ulimit memlock=-1 -v /home/$USER/NeutrinoML/docker:/scratch --workdir /scratch nvidia/cuda:10.2-devel
