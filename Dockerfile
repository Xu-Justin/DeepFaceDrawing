FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -y git vim libgtk2.0-dev && \
    rm -rf /var/cache/apk/*

COPY requirements.txt /workspace
RUN pip --no-cache-dir install -r /workspace/requirements.txt
RUN pip --no-cache-dir install comet_ml
RUN pip --no-cache-dir install dvc[gdrive]