#!/bin/bash

# 移除已有的容器
docker rm -f foundationpose

# 设置工作目录
DIR=$(pwd)/../

# 允许 X11 连接
if ! xhost +; then
    echo "Failed to run xhost +"
    exit 1
fi

# 运行 Docker 容器
docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name foundationpose \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --privileged=true \
    -v $DIR:$DIR -v /dev:/dev -v /home:/home -v /mnt:/mnt -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp:/tmp -v /etc/udev:/etc/udev -v /run/udev:/run/udev --ipc=host \
    -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE foundationpose:latest bash -c "cd $DIR && bash"