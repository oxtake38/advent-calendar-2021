FROM python:3.8.12
RUN pip install tensorflow==2.7.0
WORKDIR /logs
ENTRYPOINT ["tensorboard", "--logdir", "/logs", "--host", "0.0.0.0"]  