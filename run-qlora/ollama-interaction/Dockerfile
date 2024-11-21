FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ENV GIT_SSL_NO_VERIFY=true

RUN apt-get update \
    && apt-get install --assume-yes --no-install-recommends ca-certificates

COPY ./sources.list /etc/apt/sources.list

RUN apt-get update
