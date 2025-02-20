# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.8.19
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install NLTK before other requirements
RUN pip install nltk

# Download NLTK data
RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data stopwords

# Give permissions
RUN chmod -R 755 /usr/local/share/nltk_data

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install Cython==0.29.32
# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Set NLTK_DATA environment variable
ENV NLTK_DATA=/usr/local/share/nltk_data
# Set environment variable for numba
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
RUN mkdir -p /tmp/numba_cache && chmod 777 /tmp/numba_cache

# Copy the source code into the container.
COPY *.py
COPY requirements.txt

# Ensure proper permissions for output directory
# Set working directory
WORKDIR /app

# Change ownership of the app directory to appuser
RUN chown -R appuser:appuser /app

# Switch to the non-privileged user to run the application.
USER appuser

# Expose the port that the application listens on.
EXPOSE 8800

# # Run the application.
# CMD python3 main.py \
#     -i data/input/mtsamples-demo.csv \
#     -o data/output/mtsamples-output.jsonl \
#     -u ../2020AB-full/2020AB-quickumls-install \
#     -t description \
#     -d id \
#     --batch-size 1
