"""Data models and schemas for API requests and responses."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime


class QueryRequest(BaseModel):
    """Request schema for query endpoint."""
    
    q: str = Field(..., description="Query string", min_length=1)
    k: int = Field(default=5, description="Number of chunks to retrieve", ge=1, le=20)
    retrieval_method: Optional[Literal["contextual", "bm25", "tfidf", "hybrid"]] = Field(
        default="hybrid",
        description="Retrieval method to use"
    )
    user_id: Optional[str] = Field(None, description="User identifier for audit logging")
    session_id: Optional[str] = Field(None, description="Session identifier")


class Citation(BaseModel):
    """Citation information for answer verification."""
    
    source_document: str = Field(..., description="Source document filename")
    page: Optional[int] = Field(None, description="Page number")
    chunk_id: str = Field(..., description="Chunk identifier")
    excerpt: str = Field(..., description="Relevant text excerpt (max 200 chars)")
    confidence: float = Field(..., description="Confidence score for this citation", ge=0.0, le=1.0)


class RetrievalSource(BaseModel):
    """Individual retrieved chunk with metadata."""
    
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., description="Chunk content")
    score: float = Field(..., description="Retrieval score")
    page: Optional[int] = Field(None, description="Page number in source document")
    source_document: Optional[str] = Field(None, description="Source document filename")
    method: str = Field(..., description="Retrieval method used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QueryResponse(BaseModel):
    """Response schema for query endpoint."""
    
    answer: str = Field(..., description="Generated answer from LLM")
    confidence_score: float = Field(..., description="Overall confidence score (0-1)", ge=0.0, le=1.0)
    confidence_level: Literal["high", "medium", "low"] = Field(..., description="Confidence level")
    citations: List[Citation] = Field(default_factory=list, description="Answer citations with sources")
    sources: List[RetrievalSource] = Field(..., description="Retrieved chunks with scores")
    query: str = Field(..., description="Original query")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    latency_ms: float = Field(..., description="Total query latency in milliseconds")
    retrieval_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Retrieval method statistics"
    )


class GroundTruthQA(BaseModel):
    """Ground truth QA pair schema."""
    
    question: str = Field(..., description="Question text")
    answer: str = Field(..., description="Ground truth answer")
    page: int = Field(..., description="Page number", ge=1)
    span_start: int = Field(..., description="Character start position")
    span_end: int = Field(..., description="Character end position")
    context: Optional[str] = Field(None, description="Surrounding context")


class BenchmarkResult(BaseModel):
    """Benchmark evaluation result."""
    
    method: str = Field(..., description="Retrieval method")
    avg_latency_ms: float = Field(..., description="Average latency")
    p95_latency_ms: float = Field(..., description="95th percentile latency")
    avg_semantic_similarity: float = Field(..., description="Average cosine similarity")
    recall_at_1: Optional[float] = Field(None, description="Recall@1")
    recall_at_3: Optional[float] = Field(None, description="Recall@3")
    recall_at_5: Optional[float] = Field(None, description="Recall@5")
    total_queries: int = Field(..., description="Number of queries evaluated")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: Literal["healthy", "unhealthy"] = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    llm_status: Optional[str] = Field(None, description="LLM connection status")
    vector_store_status: Optional[str] = Field(None, description="Vector store status")


class AuditLog(BaseModel):
    """Audit log entry for compliance and security."""
    
    log_id: str = Field(..., description="Unique audit log identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of event")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    query: str = Field(..., description="User query")
    answer: str = Field(..., description="Generated answer")
    confidence_score: float = Field(..., description="Confidence score")
    documents_accessed: List[str] = Field(default_factory=list, description="List of documents accessed")
    retrieval_method: str = Field(..., description="Retrieval method used")
    latency_ms: float = Field(..., description="Query latency in milliseconds")
    cache_hit: bool = Field(..., description="Whether result came from cache")
    success: bool = Field(default=True, description="Whether query succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
