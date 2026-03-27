# ===== Base image =====
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    # ensure conda uses our condarc from the first call
    CONDARC=/root/.condarc

# ===== System deps =====
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget curl bzip2 ca-certificates git tini bash coreutils gawk \
      build-essential hydra tmux \
 && rm -rf /var/lib/apt/lists/*

# ===== Miniconda =====
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/conda.sh \
 && bash /tmp/conda.sh -b -p "$CONDA_DIR" \
 && rm /tmp/conda.sh

# ===== Write a conda-forge-only condarc BEFORE touching conda =====
# This prevents any access to repo.anaconda.com/pkgs/* (and avoids ToS prompts)
RUN printf '%s\n' \
  'channels:' \
  '  - conda-forge' \
  'default_channels:' \
  '  - https://conda.anaconda.org/conda-forge' \
  'custom_multichannels:' \
  '  defaults: []' \
  'channel_priority: strict' \
  'show_channel_urls: true' \
  > /root/.condarc

# Optional: also drop any vendor condarc that might be shipped in $CONDA_DIR
RUN rm -f "$CONDA_DIR/.condarc" || true

# ===== Base conda tooling from conda-forge =====
RUN conda install -n base -y conda-libmamba-solver mamba \
 && conda config --set solver libmamba \
 && conda update -n base -y conda

# ===== Bring in any extra specs you might still want =====
# (environment.yml is optional; it's not used directly below but harmless to keep)
COPY environment.from_history.yml /tmp/environment.yml

# ===== Bring in the loose pip requirements (no version pins) =====
COPY requirements.loose.txt /tmp/requirements.loose.txt

SHELL ["/bin/bash", "-lc"]

# ===== Create the env with Python 3.11 and system GDAL via conda =====
RUN mamba create -y -n deshadow python=3.11 \
 && mamba install -y -n deshadow gdal \
 && conda clean -afy

# ===== Install Python packages with pip from the loose requirements =====
RUN source "$CONDA_DIR/etc/profile.d/conda.sh" \
 && conda activate deshadow \
 && python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.loose.txt

# ===== Auto-activate env & set LD_LIBRARY_PATH for interactive shells =====
RUN printf '%s\n' \
 'source "$CONDA_DIR/etc/profile.d/conda.sh"' \
 'conda activate deshadow' \
 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"' \
 > /etc/profile.d/conda_env.sh

WORKDIR /app
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["/bin/bash"]

# If you need a specific torchvision/torch combo or GPU wheels later, you can
# add another RUN with explicit pins, e.g.:
# RUN source "$CONDA_DIR/etc/profile.d/conda.sh" \
#  && conda activate deshadow \
#  && pip install --no-cache-dir \
#       torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
#       --index-url https://download.pytorch.org/whl/cpu
