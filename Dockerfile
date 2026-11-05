FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    FASTEMBED_CACHE_PATH=/home/user/app/data/model_cache/fastembed \
    HF_HOME=/home/user/app/data/model_cache/huggingface \
    INTEGRITY_MODE=strict

WORKDIR $HOME/app

# Install the rest of the backend dependencies
COPY --chown=user backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake the ONNX embedding model used by both ingestion and serving.
RUN mkdir -p "$FASTEMBED_CACHE_PATH" "$HF_HOME" && python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2', threads=1)"
RUN python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2', threads=1, providers=['CPUExecutionProvider'])"

# Copy the backend application code (including the committed vector store)
COPY --chown=user backend/ .

# Rebuild and validate the index from the shipped authorized sources at image build time.
RUN python -m app.rag.ingest && python verify_sources.py --check-index

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
