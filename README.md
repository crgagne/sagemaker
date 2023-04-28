### How to launch python jobs on Sagemaker w/ Custom Docker

(0) check the region for aws cli `aws configure get region`

(1) build docker image & upload to Amazon's ECR using `cd container` and `./build_and_push.sh pytorch-extending-our-containers-bark`

(2) test locally `docker run pytorch-extending-our-containers-bark:latest`
(2a) test locally with aws creds `docker run -v $HOME/.aws/credentials:/root/.aws/credentials:ro pytorch-extending-our-containers-bark:latest`

*Note: you can repeat steps 1 and 2 to edit the docker

(3) test on sagemaker instances using this notebook

(requires a conda environment with sagemarker installed)


Notes:
- docker containers are stored here: https://us-east-1.console.aws.amazon.com/ecr/repositories/private/064311914016/pytorch-extending-our-containers-bark?region=us-east-1
- training jobs can be monitored here: https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs