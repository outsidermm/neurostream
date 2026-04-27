# Base image: Ubuntu 24.04 LTS, multi-arch (pulls arm64 on Apple Silicon).
# Python 3.12 ships natively, so no PPA needed.
FROM ubuntu:24.04

# Environment variables baked into every shell session in the container
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:/usr/local/bin:${PATH}"

# Install OS packages in a single RUN to avoid stale-cache bugs.
# Single phase: Python 3.12 is in the default Ubuntu 24.04 repos.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        git-lfs \
        sudo \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        build-essential \
        cmake \
        ninja-build \
        clang-18 \
        lld-18 \
        pkg-config \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user matching default host UID/GID (1000:1000 on most systems).
# NOPASSWD sudo lets you apt-get install forgotten packages without rebuild.
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME

# Install uv from a pinned version. Avoid `curl | sh` because that pulls
# whatever version is current at build time, breaking reproducibility.
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /usr/local/bin/

# Pre-create the venv directory with correct ownership so `uv sync` works
# when run as the vscode user later.
RUN mkdir -p /opt/venv && chown $USERNAME:$USERNAME /opt/venv

# Switch to non-root for everything from here on.
USER $USERNAME
WORKDIR /workspaces/neurostream

# Copy only dependency files first — these change rarely.
# Source code is bind-mounted at runtime, not baked into the image.
COPY --chown=$USERNAME:$USERNAME pyproject.toml uv.lock* ./

# Pre-install dependencies into /opt/venv at build time.
# --frozen: refuse to update the lockfile (reproducibility guarantee)
# --no-install-project: skip installing the project itself; only deps
RUN uv sync --frozen --no-install-project