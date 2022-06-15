FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -y git vim libgtk2.0-dev zip unzip && \
    rm -rf /var/cache/apk/*

COPY requirements.txt /
RUN pip --no-cache-dir install -r /requirements.txt
RUN pip --no-cache-dir install comet_ml
RUN pip --no-cache-dir install dvc[gdrive]