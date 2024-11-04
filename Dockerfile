FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]

ARG UID
ARG GID
ARG GROUPNAME
RUN groupadd -g ${GID} ${GROUPNAME} && useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USERNAME}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install --assume-yes --no-install-recommends ca-certificates

# COPY sources.list /etc/apt/sources.list

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    nano \
    curl \
    wget \
    gnupg2 \
    lsb-release \
    ubuntu-keyring \
    && curl https://nginx.org/keys/nginx_signing.key | gpg --dearmor \
    | tee /usr/share/keyrings/nginx-archive-keyring.gpg >/dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/nginx-archive-keyring.gpg] http://nginx.org/packages/ubuntu `lsb_release -cs` nginx" \
    | tee /etc/apt/sources.list.d/nginx.list \
    && echo -e "Package: *\nPin: origin nginx.org\nPin: release o=nginx\nPin-Priority: 900\n" \
    | tee /etc/apt/preferences.d/99nginx \
    && apt-get update \
    && apt-get upgrade -y

# Install Python with 3.10 version, install Pyenv and activate python virtual environment

WORKDIR $HOME

COPY . .

RUN python --version \
    && apt-get install make build-essential libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncursesw5-dev \
    xz-utils tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && curl https://pyenv.run | bash \
    && export PYENV_ROOT="$HOME/.pyenv" \
    [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH" \
    eval "$(pyenv init -)" \
    && source ~/.bashrc \
    && pyenv local 3.10.15 \
    && python --version \
    && apt-get update \
    && apt-get -y upgrade
