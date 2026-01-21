import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from app.services.search_service import get_search_service, SearchService

load_dotenv()

app = FastAPI(title="Semantic Search API")

class SearchRequest(BaseModel):
    query: str

class SearchResultItem(BaseModel):
    id: str
    text_snippet: str
    score: float

@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok"}

@app.post("/search", response_model=List[SearchResultItem], status_code=200)
async def semantic_search_endpoint(
    request: SearchRequest, 
    service: SearchService = Depends(get_search_service)
):
    if not request.query or len(request.query) < 3:
        raise HTTPException(
            status_code=400, 
            detail="Query must not be empty and at least 3 characters long."
        )
    
    try:
        top_k = int(os.getenv("TOP_K_RESULTS", 5))
        results = service.search_documents(request.query, top_k=top_k)
        return results
    except Exception as e:
        # In production, log the error
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
