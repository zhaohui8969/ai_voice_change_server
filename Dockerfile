FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

ARG apt_mirror="mirrors.aliyun.com"

WORKDIR /code

ADD requirements.txt .
#RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple -r requirements.txt
RUN pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
RUN pip install -r requirements.txt
RUN chmod 777 /tmp

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update -y
RUN apt-get install libsndfile1 -y

ADD . .

EXPOSE 7780

ENTRYPOINT ["bash", "predict.sh"]
