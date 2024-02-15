# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get -y upgrade --only-upgrade systemd openssl cryptsetup \
    && apt-get install -y \
    bzip2 \
    curl \
    git \
    git-lfs \
    tar \
    vim \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# Install Poetry
RUN pip install --upgrade pip && \
    pip install poetry

# Copy only the files necessary for Poetry to install dependencies
COPY pyproject.toml /home/user/pyproject.toml
COPY poetry.lock /home/user/poetry.lock

# Configure Poetry
# Disable virtualenv creation by Poetry since Docker itself provides isolation
RUN poetry config virtualenvs.create false

# Install the project dependencies
RUN poetry install --no-dev

COPY aihero /home/user/aihero

# Set the working directory in the container to /aihero
WORKDIR /home/user/aihero

# Run peft.py when the container launches
CMD ["python", "/home/user/aihero/sft.py"]
