FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
MAINTAINER Oleg Boiko <oboiko@chegg.com>

RUN apt-get update
RUN apt-get install -y gcc zip

WORKDIR /opt/data
COPY data/ .
RUN tar -xf Oxford-2019.tar
#RUN unzip glove.6B.zip

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /