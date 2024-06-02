FROM nvcr.io/nvidia/pytorch:23.06-py3

WORKDIR /home/3d-microstructure-generation

# Python packages
RUN pip install -U pip
RUN pip install -U setuptools
RUN pip freeze > /tmp/requirements-base-image.txt
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Clean working directory
RUN rm -rf /workspace/*