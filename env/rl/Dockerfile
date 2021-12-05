FROM stablebaselines/stable-baselines3-cpu:latest

WORKDIR /work
COPY requirements.txt /work

RUN conda update conda	
RUN conda install -y conda-build
RUN pip install --upgrade --no-deps --force-reinstall stable-baselines3
RUN pip install -r requirements.txt
# RUN apt-get update -y && apt-get install -y openssh-client 