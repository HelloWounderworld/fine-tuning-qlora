FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

# ARG UID
# ARG GID
# ARG GROUPNAME
# RUN groupadd -g ${GID} ${GROUPNAME} && useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USERNAME}

ARG DEBIAN_FRONTEND=noninteractive
ENV GIT_SSL_NO_VERIFY=true
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONUTF8=1
ENV PIP_NO_CACHE_DIR=off

# COPY sources.list /etc/apt/sources.list

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install --assume-yes --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    nano \
    curl \
    wget \
    gnupg2 \
    lsb-release \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl \
    libopenblas-base \
    libopenblas-dev \
    logrotate \
    less \
    sudo \
    apt-utils \
    dpkg

# set locale to ja_JP
# ENV LANG=ja_JP.UTF-8
# ENV LANGUAGE=ja_JP:ja
# ENV TZ Asia/Tokyo
# RUN update-locale LANG=ja_JP.UTF-8 \
#     && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
#     && echo > /etc/timezone

# Install Ollama
# RUN chmod 777 /opt \
#     && curl -fsSL https://ollama.com/install.sh | sh

# Install Python with 3.10 version, install Pyenv or activate python virtual environment

WORKDIR /opt

ENV PYENV_ROOT /opt/.pyenv
ENV PATH ${PYENV_ROOT}/bin:${PYENV_ROOT}/shims:${PATH}
ARG PYTHON_VERSION=3.10.15
RUN git clone https://github.com/pyenv/pyenv.git ${PYENV_ROOT} \
    && echo 'export PATH="${PYENV_ROOT}/bin:${PYENV_ROOT}/shims:${PATH}"' >> ~/.bashrc \
    && echo 'eval "$(pyenv init -)"' >> ~/.bashrc \
    && eval "$(pyenv init -)" \
    && source ~/.bashrc \
    && CFLAGS=-I/usr/include LDFLAGS=-L/usr/lib pyenv install -v ${PYTHON_VERSION} \
    && pyenv global ${PYTHON_VERSION} \
    && pyenv rehash \
    && python --version \
    && pip install --upgrade pip

# Install CUDA
WORKDIR /teramatsu/qlora

COPY . .

RUN apt-get update \
    && apt-get -y upgrade \
    && nvcc --version \
    && python -m venv .venv \
    && source .venv/bin/activate \
    && pip install -U -r requirements.txt \
    && pip freeze > requirements.txt

EXPOSE 8000

CMD nvidia-smi
