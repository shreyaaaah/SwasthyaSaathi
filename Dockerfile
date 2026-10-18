FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install PyTorch CPU-only build first (smaller than the default CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the backend dependencies
COPY --chown=user backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake both sentence-transformer models into the image so the first
# request doesn't need to download them from Hugging Face Hub at runtime.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy the backend application code (including the committed vector store)
COPY --chown=user backend/ .

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
