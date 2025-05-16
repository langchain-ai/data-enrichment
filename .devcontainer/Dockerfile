FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s $HOME/.local/bin/poetry /usr/local/bin/poetry

EXPOSE 14500
