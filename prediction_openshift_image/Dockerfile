FROM ubuntu:20.04
#FROM python:3.7

# Here some optional proxy configurations

ENV PYTHONUNBUFFERED=1
# Allow python script to import packages at "/pipelines/component/src/"
ENV PYTHONPATH="/pipelines/component/src/:$PYTHONPATH"

# Autoclean does not really reduce the image size
# Probably due to a lot of dependencies
RUN apt-get update && apt-get install -y \
        python3-pip vowpal-wabbit && \
	apt-get clean && apt-get autoclean

# Python dependencies
COPY requirements.txt .
RUN  pip3 install --upgrade pip && pip3 install -r requirements.txt

# Copy model and files into container image
# COPY model /pipelines/component/src/model
COPY function /pipelines/component/src
RUN mkdir /pipelines/component/cwd
RUN chmod 777 /pipelines/component/cwd
# RUN mkdir /pipelines/component/cwd/model
