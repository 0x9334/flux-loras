FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install git git-lfs wget -y && apt-get clean

RUN mkdir -p ~/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh \
    && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
    && rm ~/miniconda3/miniconda.sh

ENV PATH="/root/miniconda3/bin:${PATH}"
WORKDIR /app
RUN conda create -n env python==3.10 -y

COPY ./requirements.txt /app
RUN conda run --no-capture-output -n env python -m pip install -r /app/requirements.txt
RUN rm /app/requirements.txt