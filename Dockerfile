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

# Copy the current directory contents into the container
COPY requirements.txt /home/user/requirements.txt
COPY requirements-dev.txt /home/user/requirements-dev.txt

WORKDIR /home/user
# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip build && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

COPY pyproject.toml /home/user/pyproject.toml
COPY src/aihero /home/user/src/aihero
RUN pip install -e .

# Run peft.py when the container launches
CMD ["python", "/home/user/aihero/research/finetuning/sft.py"]
