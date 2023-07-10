FROM ubuntu:20.04

# Here some optional proxy configurations

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
        python3-pip && \
	apt-get clean && apt-get autoclean

RUN  pip3 install --upgrade pip
RUN  pip3 install pandas

ENTRYPOINT ["tail", "-f", "/dev/null"]




 