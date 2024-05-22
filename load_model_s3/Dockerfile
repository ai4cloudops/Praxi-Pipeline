FROM ubuntu:20.04

# Here some optional proxy configurations

ENV PYTHONUNBUFFERED=1

# Autoclean does not really reduce the image size
# Probably due to a lot of dependencies
RUN apt-get update && apt-get install -y \
        python3-pip && \
	apt-get clean && apt-get autoclean

# # Python dependencies
COPY requirements.txt .
RUN  pip3 install --upgrade pip && pip3 install -r requirements.txt
RUN  mkdir -p /pipelines/component/src
RUN  chmod 777 /pipelines/component/src
# RUN  useradd newuser
# RUN  chown newuser /pipelines/component/src
# RUN  chmod 755 /pipelines/component/src
# USER newuser
