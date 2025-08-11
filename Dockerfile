FROM nvcr.io/nvidia/pytorch:25.06-py3

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl gnupg tmux && \
    echo 'deb [trusted=yes] https://apt.fury.io/ascii-image-converter/ /' \
        > /etc/apt/sources.list.d/ascii-image-converter.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ascii-image-converter && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir jupyterlab

WORKDIR /workspace
EXPOSE 8888

CMD ["/bin/bash"]