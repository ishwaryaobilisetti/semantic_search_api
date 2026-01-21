from fastapi.testclient import TestClient
import sys
import os
import pytest
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.main import app
from app.services.search_service import SearchService, get_search_service

client = TestClient(app)

# Mock SearchService to avoid loading actual models during basic API tests
# or for full integration, use the actual one if data exists.
# Here we mock it to test the API layer logic.

def mock_get_search_service():
    mock_service = MagicMock()
    mock_service.search_documents.return_value = [
        {"id": "test_1", "text_snippet": "This is a test", "score": 0.9}
    ]
    return mock_service

@pytest.fixture
def override_dependency():
    app.dependency_overrides[get_search_service] = mock_get_search_service
    yield
    app.dependency_overrides = {}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_search_valid(override_dependency):
    response = client.post("/search", json={"query": "test query"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == "test_1"

def test_search_invalid_short():
    response = client.post("/search", json={"query": "hi"})
    assert response.status_code == 400

def test_search_invalid_empty():
    response = client.post("/search", json={"query": ""})
    assert response.status_code == 400
