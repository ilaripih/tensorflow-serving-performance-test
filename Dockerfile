FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    python3-pip

RUN pip3 install tensorflow-serving-api==1.14.0