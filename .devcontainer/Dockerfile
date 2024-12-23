FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04
LABEL maintainer="Climavision"

ARG DEBIAN_FRONTEND=noninteractive
ARG PROJ=Aurora
RUN apt-get update

# Install general packages
RUN apt-get install -y --fix-missing \
    bzip2 \
    ca-certificates \
    curl \
    ffmpeg \
    gcc \
    git \
    libcurl4-openssl-dev \
    libffi-dev \
    libfreetype6-dev \
    libgl1-mesa-glx \
    libjpeg-dev \
    libopenjp2-7-dev \
    libssl-dev \
    libtiff5-dev \
    python3-pip \
    python3-venv \
    r-base \ 
    software-properties-common \
    supervisor \
    vim \
    wget \ 
    zlib1g \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN update-ca-certificates -f

# Create a non-root user and activate it
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME


# Install Mambaforge as root
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/mambaforge.sh && \
    bash ~/mambaforge.sh -b -p /opt/mambaforge && \
    rm ~/mambaforge.sh && \
    chown -R $USERNAME:$USERNAME /opt/mambaforge  # change ownership to $USERNAME user

# Set path to mamba
ENV PATH /opt/mambaforge/bin:$PATH

COPY environment.yml environment.yml
COPY requirements-dev.txt requirements-dev.txt

# Create conda environment and install python dependencies
RUN mamba env create -f environment.yml && \
    mamba clean -ya

# Activate the environment.  Need to do this properly, otherwise
# some packages won't get initialized quite right. Fiona is a good example
# of this.
RUN echo "mamba activate ${PROJ}" > ~/.bashrc
ENV PATH /opt/mambaforge/envs/$PROJ/bin:$PATH
ENV CONDA_DEFAULT_ENV $PROJ

RUN python3 -m pip install -r requirements-dev.txt

# This will make VS Code happy for development

USER $USERNAME
ENV SHELL /bin/bash
CMD [ "/bin/bash" ]
