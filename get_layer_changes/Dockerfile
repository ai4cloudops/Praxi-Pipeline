FROM ubuntu:20.04

# Here some optional proxy configurations

ENV PYTHONUNBUFFERED=1

# Autoclean does not really reduce the image size
# Probably due to a lot of dependencies
RUN apt-get update && apt-get install -y \
        curl jq python3-pip && \
	apt-get clean && apt-get autoclean

# # Python dependencies
COPY requirements.txt .
RUN  pip3 install --upgrade pip && pip3 install -r requirements.txt


# Allow python script to import packages at "/pipelines/component/src/"
ENV PYTHONPATH="/pipelines/component/src/:$PYTHONPATH"

# Copy into container image
COPY src /pipelines/component/src
RUN mkdir /pipelines/component/cwd
RUN chmod 777 /pipelines/component/cwd
RUN mkdir /pipelines/component/cwd/changesets
RUN chmod 777 /pipelines/component/cwd/changesets