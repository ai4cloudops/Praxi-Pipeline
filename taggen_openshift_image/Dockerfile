FROM python:3.7

# Here some optional proxy configurations

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/pipelines/component/src/:$PYTHONPATH"
# Autoclean does not really reduce the image size
# Probably due to a lot of dependencies
RUN apt-get update && apt-get install -y &&\
	apt-get clean && apt-get autoclean

# Python dependencies
COPY requirements.txt .
RUN  pip3 install --upgrade pip && pip3 install -r requirements.txt

COPY columbus /pipelines/component/src/columbus
COPY tagset_gen.py /pipelines/component/src
RUN mkdir /pipelines/component/cwd
RUN chmod 777 /pipelines/component/cwd
# COPY function /pipelines/component/src/function

# COPY mix_test_tag/mix_test_tag /pipelines/component/src/mix_test_tag
# COPY ml_train_tag/ml_train_tag /pipelines/component/src/ml_train_tag 
# COPY demo_tagsets_mostly_single_label/demo_tagsets /pipelines/component/src/demo_tagsets

# # Add the vwserver user
# RUN mkdir /home/vwserver
# RUN useradd vwserver -s /bin/bash -u 1001 \
#     -d /home/vwserver 
# RUN  chown 1001:1001 -R /home/vwserver 
# RUN  mkdir /vw_models && \
#      chown 1001:1001 -R /vw_models && \
#      chmod u+rw -R /vw_models

# # Transfer serving libraries and entrypoint
# COPY --chown=vwserver flaskserver /home/vwserver/flaskserver
# COPY --chown=vwserver vwmodel /home/vwserver/vwmodel
# COPY --chown=vwserver docker/start_server.sh /home/vwserver

# ENTRYPOINT ["/home/vwserver/start_server.sh"]



# docker run -it -u 1001:1001 -v $(pwd)/localVolume:/vw_models zongshun96/vwserver:0.01 /bin/bash
# docker run -it -u 1001:1001 -v $(pwd)/localVolume:/vw_models -p 6025:6025 -e MODEL_PORT="6025" zongshun96/vwserver:0.01