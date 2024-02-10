# docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

RUN groupadd --gid 1000 node \
  && useradd --uid 1000 --gid node --shell /bin/bash --create-home node

RUN  apt-get update -yq \
  && apt-get install curl gnupg -yq 
#   && curl -sL https://deb.nodesource.com/setup_12.x | bash \
#  && apt-get install nodejs -yq

RUN apt-get update  && \
  apt-get install -y --no-install-recommends \
  git \
  libjpeg-dev \
  libsm6 \
  libxext6 \
  ffmpeg \
  zlib1g-dev \
  google-perftools && \
  apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


#Install extension in Jupyter lab
RUN pip3 install jupyter jupyterlab 


RUN pip3 install -U pip


COPY requirements.txt /tmp/requirements.txt

RUN cd /tmp/ && \
  pip install  gdown==4.3.0  \
  sacred==0.8.2 \
  scipy==1.7.3 \
  matplotlib \
  rasterio \
  opencv-python \
  tqdm \
  pandas \
  scikit-learn==1.0.2 \
  albumentations[imgaug]==1.1.0 \
  segmentation-models-pytorch==0.3.0


EXPOSE 8888

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]
