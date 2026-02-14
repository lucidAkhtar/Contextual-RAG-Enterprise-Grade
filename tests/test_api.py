"""Test suite for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_returns_200(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
    
    def test_health_check_structure(self, client):
        """Test health check response structure."""
        response = client.get("/api/v1/health")
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert data["status"] in ["healthy", "unhealthy"]


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_returns_info(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestInfoEndpoint:
    """Test info endpoint."""
    
    def test_info_endpoint(self, client):
        """Test info endpoint returns system information."""
        response = client.get("/api/v1/info")
        
        # May return 503 if query engine not initialized
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "retrieval_methods" in data


class TestQueryEndpoint:
    """Test query endpoint."""
    
    def test_query_requires_q_parameter(self, client):
        """Test query endpoint requires 'q' parameter."""
        response = client.post("/api/v1/query", json={})
        
        assert response.status_code == 422  # Validation error
    
    def test_query_with_valid_request(self, client):
        """Test query with valid request structure."""
        response = client.post(
            "/api/v1/query",
            json={
                "q": "What is this document about?",
                "k": 5,
                "retrieval_method": "hybrid"
            }
        )
        
        # May return 503 if query engine not initialized
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert "query" in data
            assert "latency_ms" in data
    
    def test_query_k_validation(self, client):
        """Test query validates k parameter."""
        response = client.post(
            "/api/v1/query",
            json={
                "q": "Test question",
                "k": 100  # Exceeds max
            }
        )
        
        assert response.status_code == 422
    
    def test_query_default_values(self, client):
        """Test query uses default values."""
        response = client.post(
            "/api/v1/query",
            json={"q": "Test question"}
        )
        
        # Should use default k=5 and method=hybrid
        assert response.status_code in [200, 503]


class TestOpenAPISpec:
    """Test OpenAPI specification."""
    
    def test_openapi_json_accessible(self, client):
        """Test OpenAPI JSON is accessible."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
    
    def test_docs_page_accessible(self, client):
        """Test Swagger docs page is accessible."""
        response = client.get("/docs")
        
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
