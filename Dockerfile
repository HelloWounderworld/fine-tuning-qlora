FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]

ARG UID
ARG GID
ARG GROUPNAME
RUN groupadd -g ${GID} ${GROUPNAME} && useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USERNAME}

ARG DEBIAN_FRONTEND=noninteractive
