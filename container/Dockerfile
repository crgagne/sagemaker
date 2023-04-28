FROM public.ecr.aws/k1e6s8o8/python:3.9-yum-latest

RUN yum -y install git
RUN pip install --upgrade pip
RUN pip install setuptools --upgrade
RUN pip install git+https://github.com/suno-ai/bark.git

COPY code/generate.py /opt/ml/generate.py


ENTRYPOINT ["/usr/bin/python3", "/opt/ml/generate.py"]