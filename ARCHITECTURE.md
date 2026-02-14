# System Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Components](#system-components)
3. [Design Decisions](#design-decisions)
4. [Data Flow](#data-flow)
5. [Scalability Considerations](#scalability-considerations)
6. [Security](#security)

## Overview

The Contextual RAG system is built using a **layered architecture** that separates concerns and enables independent scaling and testing of components.

### Architecture Principles

1. **Separation of Concerns**: Clear boundaries between API, business logic, and data layers
2. **Dependency Injection**: Components receive dependencies explicitly, improving testability
3. **Interface-Based Design**: Abstract base classes enable easy component replacement
4. **Configuration Management**: Centralized settings with environment variable support
5. **Observability**: Comprehensive logging and metrics collection throughout

## System Components

### 1. API Layer (`src/api/`)

**Responsibility**: Handle HTTP requests, input validation, and response formatting

**Components**:
- `routes.py`: REST endpoint definitions
- FastAPI framework for OpenAPI spec generation
- Pydantic models for request/response validation

**Design Patterns**:
- **MVC Pattern**: Routes act as controllers, schemas as models
- **Dependency Injection**: Query engine injected into routes

```python
# API receives query -> validates -> delegates to engine -> formats response
POST /query → validate(QueryRequest) → engine.query() → format(QueryResponse)
```

### 2. Core Engine (`src/core/`)

**Responsibility**: Orchestrate RAG pipeline from ingestion to answer generation

#### 2.1 Query Engine (`query_engine.py`)

**Role**: Facade for the entire RAG pipeline

**Responsibilities**:
- Document ingestion and indexing
- Retrieval method coordination
- LLM-based answer generation
- Confidence scoring and citation extraction
- Performance metric collection

**Key Methods**:
```python
def ingest_document(pdf_path) → void
    """Load PDF → Chunk → Index → Initialize retrievers"""

def query(query_text, top_k, method) → (answer, sources, confidence, citations, stats)
    """Retrieve → Generate → Calculate Confidence → Extract Citations → Measure"""

def calculate_confidence(sources) → (score, level)
    """Algorithm: 70% top score + 20% avg score + 10% consistency
    Returns: score (0-1) and level (high/medium/low)
    Thresholds: high (≥75%), medium (50-75%), low (<50%)"""

def create_citations(sources, max_citations=3) → list[Citation]
    """Extract top sources with document, page, chunk ID, and 200-char excerpts
    Normalizes scores to confidence values for answer traceability"""
```

#### 2.2 Document Processor (`document_processor.py`)

**Role**: PDF extraction and preprocessing

**Features**:
- Multi-page PDF parsing with PyMuPDF
- Table extraction and formatting
- Text cleaning and normalization
- Metadata preservation (page numbers, source)

**Key Innovation**: Preserves table structure while making it searchable

#### 2.3 Chunking Strategies (`chunking_strategies.py`)

**Role**: Convert documents into retrievable chunks

**Strategy Pattern Implementation**:

```
ChunkingStrategy (ABC)
├── FixedSizeChunkingStrategy
│   └── Traditional fixed-size with overlap
├── SemanticChunkingStrategy
│   └── Boundary detection via embedding similarity
└── SentenceChunkingStrategy
    └── Sentence-boundary aware chunking
```

**Factory Pattern**:
```python
strategy = ChunkingStrategyFactory.create_strategy("fixed_size")
nodes = strategy.chunk_documents(documents)
```

#### 2.4 Retrievers (`retrievers.py`)

**Role**: Find relevant chunks for queries

**Four Retrieval Methods**:

1. **Contextual Retriever** (Anthropic's Approach)
   ```
   For each chunk:
     1. Generate context using LLM: "This chunk discusses..."
     2. Prepend context to chunk
     3. Embed enriched chunk
     4. Store in vector database
   
   At query time:
     1. Embed query
     2. Vector similarity search
     3. Return top-k chunks
   ```

2. **BM25 Retriever** (Lexical)
   ```
   - Tokenize corpus
   - Build inverted index
   - Score using BM25 formula:
     score = Σ IDF(q_i) * (f(q_i, D) * (k1 + 1)) / 
                           (f(q_i, D) + k1 * (1 - b + b * |D| / avgdl))
   ```

3. **TF-IDF Retriever** (Statistical)
   ```
   - Build term frequency matrix
   - Calculate IDF weights
   - Compute cosine similarity
   ```

4. **Hybrid Retriever** (Fusion)
   ```
   Reciprocal Rank Fusion (RRF):
   
   score(doc) = Σ_method [ weight(method) / (k + rank_method(doc)) ]
   
   where k = 60 (RRF constant)
   ```

**Comparison**:

| Method | Strength | Weakness | Use Case |
|--------|----------|----------|----------|
| Contextual | Semantic understanding | Slow indexing | Best quality |
| BM25 | Fast, keyword-focused | Misses synonyms | Exact terms |
| TF-IDF | Fast, statistical | Lacks semantics | Frequency-based |
| Hybrid | Best of all | Most complex | Production default |

#### 2.5 LLM Factory (`llm_factory.py`)

**Role**: Abstract LLM provider differences

**Factory Pattern**:
```python
LLMFactory
├── create_provider(name) → LLMProvider
└── get_llm(name) → LLM

LLMProvider (ABC)
├── OllamaProvider
├── OpenAIProvider
└── [Extensible to Anthropic, Cohere, etc.]
```

**Benefits**:
- Switch providers via configuration
- Consistent interface across providers
- Easy to add new providers

### 3. Models Layer (`src/models/`)

**Responsibility**: Data validation and type safety

**Pydantic Models**:
- `QueryRequest`: API input validation (includes user_id, session_id for audit)
- `QueryResponse`: Structured API output (includes confidence_score, confidence_level, citations)
- `Citation`: Source attribution with document, page, excerpt, and confidence
- `AuditLog`: Complete audit trail with user, IP, documents accessed, confidence, and timing
- `GroundTruthQA`: Benchmark data schema
- `BenchmarkResult`: Evaluation metrics

**Benefits**:
- Automatic validation
- Type hints for IDE support
- OpenAPI schema generation

### 4. Utilities (`src/utils/`)

**Cross-cutting concerns**:

- **Logger**: Structured logging with file and console outputs
- **Metrics Collector**: Performance monitoring (Observer pattern)
- **Evaluation Functions**: Cosine similarity, Recall@K calculation

### 5. Storage Layer (`src/core/cache_store.py`)

**Responsibility**: Query caching and audit logging using TinyDB

**Key Features**:
- **Cache Table**: Stores query results with TTL for performance optimization
- **Audit Table**: Enterprise audit trail with full query lifecycle tracking
- **Audit Methods**:
  - `add_audit_log()`: Records every query with user_id, IP address, confidence score, documents accessed, latency, and success status
  - `get_audit_logs()`: Retrieves logs with filtering (user, date range, success)
  - `get_audit_statistics()`: Compliance reporting (total queries, success rate, unique users, average confidence)

**Production Design**: TinyDB provides production-like persistence with atomic operations, suitable for migration to MongoDB/PostgreSQL

## Design Decisions

### 1. Why LlamaIndex?

**Chosen over LangChain/LangGraph**:
-  Lower memory footprint (~300MB vs ~600MB)
-  Purpose-built for RAG workflows
-  Cleaner abstractions for retrieval
-  Better documentation for production use

### 2. Why ChromaDB?

**Chosen over FAISS/Qdrant**:
-  Embedded database (no separate server needed)
-  Persistence out of the box
-  Ideal for development and demos
-  Easy migration path to production (Qdrant/Pinecone)

### 3. Why FastAPI?

**Chosen over Flask/Django**:
-  Automatic OpenAPI specification
-  Built-in validation with Pydantic
-  Async support for high concurrency
-  Modern Python (type hints, async/await)

### 4. Why Strategy Pattern for Chunking?

**Benefits**:
- Easy to test different chunking approaches
- No code changes to switch strategies
- Extensible to custom strategies
- Each strategy encapsulates its algorithm

### 5. Why Contextual Enrichment?

**Based on Anthropic's Research**:
- Improves retrieval accuracy by 20-30%
- Provides document-level context to chunks
- Minimal impact on query latency (only indexing slower)
- Worth the tradeoff for quality

## Data Flow

### Ingestion Flow

```
PDF File
   ↓
Document Processor (PyMuPDF)
   ↓
Pages with Metadata
   ↓
Chunking Strategy
   ↓
Text Nodes
   ↓
Contextual Enrichment (LLM)
   ↓
Embedding Model
   ↓
Vector Store (ChromaDB)
   
Parallel:
   ↓
BM25 Index
   ↓
TF-IDF Index
```

### Query Flow

```
User Query
   ↓
API Layer (Validation)
   ↓
Query Engine
   ↓
Retrieval (Parallel)
   ├── Contextual: Vector Search
   ├── BM25: Lexical Ranking
   └── TF-IDF: Statistical Ranking
   ↓
Hybrid Fusion (RRF)
   ↓
Top-K Chunks
   ↓
Confidence Calculation
   ↓
Citation Extraction
   ↓
LLM (Answer Generation)
   ↓
Cache Storage + Audit Logging
   ↓
Response + Sources + Confidence + Citations
   ↓
API Response (JSON)
```

### Benchmark Flow

```
Ground Truth QA Pairs
   ↓
For each query:
   ├── Execute query
   ├── Measure latency
   ├── Calculate semantic similarity
   └── Calculate Recall@K
   ↓
Aggregate Metrics
   ↓
Generate Report (Markdown + JSON)
```

## Scalability Considerations

### Current Architecture (Single Node)

**Suitable for**:
- Development and testing
- POC demonstrations
- Up to ~1000 concurrent users
- Document corpus up to ~10GB

**Bottlenecks**:
- Single LLM instance
- In-process embedding computation
- Single vector database

### Production Scaling Path

#### 1. Horizontal Scaling (Stateless API)

```
Load Balancer
   ├── API Instance 1 ──┐
   ├── API Instance 2 ──┼── Shared ChromaDB (Network Mode)
   └── API Instance N ──┘
```

#### 2. Distributed Vector Store

```
Replace ChromaDB with:
- Qdrant (self-hosted cluster)
- Pinecone (managed service)
- Weaviate (hybrid cloud)
```

#### 3. Separate Embedding Service

```
API Instances → Embedding Service (GPU) → Vector Store
                      ↑
                Batching + Caching
```

#### 4. LLM Optimization

**Options**:
- Multiple Ollama instances (load balanced)
- vLLM for batched inference
- LLM provider API (OpenAI, Anthropic) for infinite scale

#### 5. Caching Layer

```
API → Redis Cache → RAG Pipeline
         ↑
   Cache query results (1 hour TTL)
```

**Expected Performance**:
- Cache hit rate: 30-40%
- Latency reduction: 90% on hits

### Estimated Scaling Numbers

| Setup | QPS | Latency (P95) | Cost/Month |
|-------|-----|---------------|------------|
| Current (1 node) | 10 | 500ms | $0 (local) |
| 5 API + Qdrant | 100 | 300ms | $500 |
| 10 API + GPU + Pinecone | 1000 | 200ms | $2000 |
| Full prod (+ caching) | 5000 | 150ms | $5000 |

## Security

### Current Implementation

-  **Authentication**: None (add JWT/OAuth for production)
-  **Input Validation**: Pydantic models
-  **CORS**: Configurable middleware
-  **Rate Limiting**: None (add Redis-based limiting)
-  **Error Handling**: No sensitive data in responses

### Production Hardening

1. **Add Authentication**
```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@router.post("/query")
async def query(request: QueryRequest, token: str = Depends(security)):
    # Validate token
    pass
```

2. **Add Rate Limiting**
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/query")
@limiter.limit("10/minute")
async def query(request: QueryRequest):
    pass
```

3. **Add Request Logging**
```python
# Log all requests for audit
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response
```

4. **Secrets Management**
- Use AWS Secrets Manager / HashiCorp Vault
- Never commit API keys to git
- Rotate keys regularly

## Testing Strategy

### Unit Tests
- Test individual components in isolation
- Mock external dependencies (LLM, vector store)
- Focus on business logic correctness

### Integration Tests
- Test component interactions
- Use test database (separate ChromaDB instance)
- Validate end-to-end flows

### Performance Tests
- Benchmark latency under load
- Memory profiling
- Identify bottlenecks

### Test Coverage Target
- Aim for 80%+ coverage
- 100% coverage on critical paths (retrieval, generation)

## Monitoring & Observability

### Metrics to Track

**Application Metrics**:
- Query latency (avg, p50, p95, p99)
- Queries per second
- Error rate
- Retrieval quality (semantic similarity)

**System Metrics**:
- CPU utilization
- Memory usage
- Disk I/O (for vector store)
- Network latency

**Business Metrics**:
- User engagement
- Query success rate
- Popular queries

### Recommended Tools

- **Metrics**: Prometheus + Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Tracing**: Jaeger or OpenTelemetry
- **Alerting**: PagerDuty or Opsgenie

---


**Document Version**: 1.0  
**Last Updated**: February 15, 2026  
**Author**: Md Marghub Akhtar
