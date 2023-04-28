FROM public.ecr.aws/k1e6s8o8/python:3.9-yum-latest

RUN yum -y install git
RUN pip install --upgrade pip
RUN pip install setuptools --upgrade
RUN pip install git+https://github.com/suno-ai/bark.git
RUN pip install boto3
RUN pip install click

COPY code/generate.py /opt/ml/generate.py
COPY code/run_generate.py /opt/ml/run_generate.py
COPY code/sentences_clean.csv /opt/ml/sentences_clean.csv

#ENV NVIDIA_VISIBLE_DEVICES all
#ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

#ENTRYPOINT ["/usr/bin/python3", "/opt/ml/generate.py"]
ENTRYPOINT ["/usr/bin/python3", "/opt/ml/run_generate.py"]