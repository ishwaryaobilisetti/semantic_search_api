# 🚀 Semantic Search API

A production-ready, containerized backend service that provides **Semantic Search** capabilities over custom datasets. Unlike traditional keyword-based search, this API understands the **intent and context** of queries using vector embeddings and high-performance similarity indexing.

---

## 🎯 Project Objective
The goal of this project is to build a robust, scalable microservice that:
1.  **Transforms** text into dense reaching vector representations (embeddings).
2.  **Indexes** these vectors using Faiss for lightning-fast similarity retrieval.
3.  **Serves** the results via a high-performance FastAPI backend.

---

## ⚡ Step-by-Step for Evaluators

Follow these commands in order to build, run, and verify the service:

### 1. Start the Service
```bash
docker-compose up --build
```
*(Once you see "Uvicorn running on http://0.0.0.0:8000", the server is ready)*

### 2. Verify Health
```bash
curl http://localhost:8000/health
```

### 3. Test Semantic Search
**For Bash/Linux/Mac:**
```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "how does artificial intelligence impact the world?"}'
```

**For PowerShell (Windows):**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/search" -Method Post -ContentType "application/json" -Body '{"query": "how does artificial intelligence impact the world?"}'
```

---

## 🏗️ Architecture & Technical Choices

### Core Stack
- **FastAPI**: Chosen for its high performance (Starlette/Pydantic), native async support, and automatic OpenAPI documentation.
- **Sentence-Transformers (`all-MiniLM-L6-v2`)**: A state-of-the-art model that balances embedding quality with extreme inference speed.
- **Faiss (`IndexFlatL2`)**: An industry-standard vector database library developed by Meta for efficient similarity search in high-dimensional spaces.
- **Docker**: Provides a consistent, isolated environment ensuring the system runs identically for every evaluator.

### Process Flow
1.  **Initialization**: On startup, `entrypoint.sh` checks for existing indices. If missing, it triggers `generate_embeddings.py`.
2.  **Encoding**: Search queries are encoded into 384-dimensional vectors.
3.  **Search**: Faiss performs an L2 distance search to find the top-K closest vectors in the index.
4.  **Response**: The API maps vector indices back to original document text and returns them with metadata.

---

## 📂 Project Structure

```text
semantic-search-api/
├── app/
│   ├── main.py                # FastAPI entry point & input validation
│   └── services/
│       └── search_service.py  # Singleton service for model/index management
├── data/                      # Persistent storage for .json and .bin files
├── models/                    # Cached transformer models
├── scripts/
│   ├── generate_embeddings.py  # Data prep & indexing logic
│   └── entrypoint.sh          # Container lifecycle management
├── tests/                     # Unit & Integration test suites
├── Dockerfile                 # Optimized Python build
├── docker-compose.yml         # Container orchestration
├── .env.example               # Environment configuration template
└── README.md
```

---

## 🚀 API Documentation

### **GET** `/health`
Check if the service and its resources (Model/Index) are healthy.
- **Status Code**: `200 OK`
- **Response**: `{"status": "ok"}`

### **POST** `/search`
Perform a semantic similarity search.
- **Request Body**:
  ```json
  {
    "query": "how does artificial intelligence impact the world?"
  }
  ```
- **Response Example**:
  ```json
  [
    {
      "id": "doc_42",
      "text_snippet": "Artificial intelligence is transforming industries...",
      "score": 0.8542
    }
  ]
  ```
- **Validation**:
  - Minimum 3 characters required.
  - Returns `400 Bad Request` for invalid/empty inputs.

---

## 🧪 Testing

The project includes a full test suite covering both unit and integration scenarios.

### **Run locally**
```bash
pytest tests/
```

### **Run in Docker**
```bash
docker exec -it sematic_search_api-web-1 pytest tests/
```

---

## 📈 Future Scalability
- **Vector DB**: Move to Milvus or Qdrant for horizontal scaling to billions of vectors.
- **Quantization**: Use ONNX with Int8 quantization to reduce latency by 2x-3x.
- **Async Indexing**: Implement a background worker (Celery/RabbitMQ) for real-time document updates.
- **Authentication**: Add JWT-based security and rate limiting.

---

## 👨‍💻 Author
Built with efficiency and scalability in mind.

