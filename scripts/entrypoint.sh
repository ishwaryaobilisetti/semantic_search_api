#!/bin/bash
set -e

# Generate embeddings if not present
if [ ! -f "data/documents.json" ] || [ ! -f "data/faiss_index.bin" ]; then
    echo "Data not found. Running generation script..."
    python scripts/generate_embeddings.py
else
    echo "Data found. Skipping generation."
fi

# Start the application
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
