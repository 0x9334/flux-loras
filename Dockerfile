FROM flux-dynamic-lora-env:latest
WORKDIR /app

ENV HF_HUB_CACHE="/app/.cache/huggingface"
RUN conda init bash
RUN echo "conda activate env" >> ~/.bashrc

COPY lora /app/lora
EXPOSE 8000
COPY entry.sh /app
COPY server.py /app
COPY stream.py /app
CMD ["bash", "entry.sh"]