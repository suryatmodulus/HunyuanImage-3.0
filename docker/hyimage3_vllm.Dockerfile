# Dockerfile of hunyuanimage3-vllm
FROM vllm/vllm-openai:v0.11.0 as base
ENTRYPOINT []

RUN ln -sf /usr/bin/python3 /usr/bin/python &&  \
    pip install --no-cache-dir git+https://github.com/huggingface/transformers && \
    git clone https://github.com/Tencent-Hunyuan/HunyuanImage-3.0 /root/HunyuanImage-3.0 && \
    pip install apache-tvm-ffi==0.1.0b15 && \
    pip install diffusers transformers accelerate && \
    pip install /root/HunyuanImage-3.0 && \
    git clone --branch feature/hunyuan_image_3.0 https://github.com/kippergong/vllm.git && \
    cd vllm && VLLM_USE_PRECOMPILED=1 pip install --editable .

RUN apt-get update && \
    apt-get install -y openssh-server && \
    mkdir /var/run/sshd && \
    apt-get install -y tmux && \
    apt-get install -y screen && \
    apt-get install -y pdsh && \
    apt-get install -y pssh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install gpustat

RUN echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc

CMD ["/usr/sbin/sshd", "-D"]
