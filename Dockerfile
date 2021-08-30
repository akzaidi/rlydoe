FROM ubuntu:18.04
# FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ARG PYTHON_VERSION=3.7
ARG PYTORCH_DEPS=cpuonly

RUN apt-get update && apt-get install -y \
	build-essential \
	cmake \
	git \
	curl \
	wget \
	unrar \
	unzip \
	xvfb \
	ca-certificates \
	libjpeg-dev \
	libpng-dev \
	libglib2.0-0 \
	libsdl2-dev \
	libosmesa6-dev \
	libglu1-mesa \
	libglu1-mesa-dev

# install Anaconda and dependencies
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
	chmod +x ~/miniconda.sh && \
	~/miniconda.sh -b -p /opt/conda && \
	rm ~/miniconda.sh && \
	ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
	echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# install patchelf
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
	&& chmod +x /usr/local/bin/patchelf

COPY . /src
WORKDIR /src

# install MuJoCo
RUN mkdir -p /root/.mujoco \
	&& wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
	&& unzip mujoco.zip -d /root/.mujoco \
	&& mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
	&& rm mujoco.zip
RUN wget -O /root/.mujoco/mjkey.txt https://www.roboti.us/mjkey.txt
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}

# add conda paths
ENV PATH=/opt/conda/bin:$PATH 

# build and activate conda environment
RUN /opt/conda/bin/conda env update -f rlex.yaml
RUN echo "conda activate rlex" >> ~/.bashrc
SHELL ["/bin/bash", "-c", "source ~/.bashrc"]
RUN conda activate rlex

# install atari roms
WORKDIR /src/rlydoe

# RUN /src/install-atari.sh
# CMD [ "/bin/bash" ]

# Install Atari ROMs.
# RUN pip3 install atari-py
RUN mkdir roms && \
	cd roms && \
	wget http://www.atarimania.com/roms/Roms.rar && \
	unrar e Roms.rar && \
	unzip ROMS.zip && \
	unzip "HC ROMS.zip" && \
	rm ROMS.zip && \
	rm "HC ROMS.zip" && \
	rm Roms.rar && \
	python3 -m atari_py.import_roms .
