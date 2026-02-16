# Contextual RAG System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10+-orange.svg)](https://www.llamaindex.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

##  Overview

An **enterprise-grade Retrieval-Augmented Generation (RAG) system** implementing [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) approach. This system demonstrates production-ready architecture with clean OOP design, comprehensive testing, and performance benchmarking.

### Key Features

-  **Contextual Retrieval**: Anthropic's approach with LLM-enriched chunk embeddings
-  **Multiple Retrieval Methods**: Contextual (semantic), BM25 (lexical), TF-IDF (statistical), Hybrid (fusion)
-  **Method Comparison**: Side-by-side comparison of all retrieval strategies via `/compare` endpoint
-  **Flexible LLM Backend**: Switch between Ollama, OpenAI via config - no code changes
-  **Dependency Injection**: FastAPI `Depends()` pattern for testable, loosely-coupled components
-  **Confidence Scoring**: Answer confidence (high/medium/low) with source citations
-  **Audit Logging**: Complete compliance trail (user, IP, documents, confidence, timing)
-  **Production-Ready**: FastAPI + OpenAPI, comprehensive logging, metrics, caching

##  Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI REST API                        │
│                       (OpenAPI Specification)                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────▼──────────┐
                    │    Query Engine      │
                    │   (Orchestrator)     │
                    └──┬────────┬────────┬─┘
                       │        │        │
         ┌─────────────▼─┐  ┌──▼───┐  ┌▼──────────────┐
         │   Contextual  │  │ BM25 │  │    TF-IDF     │
         │   Retriever   │  │      │  │               │
         └───────┬───────┘  └──┬───┘  └───────┬───────┘
                 │             │              │
                 └─────────┬───┴──────────────┘
                           │
                  ┌────────▼────────┐
                  │  Hybrid Fusion  │
                  │  (RRF Ranking)  │
                  └────────┬────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼─────┐    ┌─────▼──────┐   ┌─────▼────┐
    │ ChromaDB │    │ Embeddings │   │   LLM    │
    │  Vector  │    │   (HFace)  │   │ (Ollama) │
    │  Store   │    │            │   │          │
    └──────────┘    └────────────┘   └──────────┘
```


##  Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running
- 8GB+ RAM recommended
- macOS, Linux, or Windows

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Contextual_RAG
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Start Ollama with Mistral model**
```bash
# In a separate terminal
ollama pull mistral
ollama serve
```

6. **Add your research paper PDF**
```bash
# Place your 8-10 page research paper PDF in:
cp your_paper.pdf data/research_paper.pdf
```

### Running the Application

#### Option 1: Quick Start (All-in-One)
```bash
# Generate ground truth and start API
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

#### Option 2: Step-by-Step

**Step 1: Generate Ground Truth QA Pairs**
```bash
python scripts/generate_ground_truth.py
```
This creates `data/ground_truth.json` with 10+ QA pairs extracted from your PDF.

**Step 2: Start the API Server**
```bash
python src/main.py
# Or with uvicorn directly:
# uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

**Step 3: Test the API**
```bash
# Check health
curl http://localhost:8000/api/v1/health

# Query the system
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "What is the main contribution of this research?",
    "k": 5,
    "retrieval_method": "hybrid"
  }'
```

**Step 4: Run Benchmarks**
```bash
python scripts/run_benchmarks.py
```
This generates `benchmarks/results.md` with comprehensive performance analysis.

##  API Documentation

Once the server is running, visit:

- **Interactive API Docs (Swagger)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### API Endpoints

#### POST `/api/v1/query`
Query the RAG system and get an answer with sources.

**Request:**
```json
{
  "q": "What is contextual retrieval?",
  "k": 5,
  "retrieval_method": "hybrid"
}
```

**Response:**
```json
{
  "answer": "Contextual retrieval is an approach that enriches document chunks...",
  "confidence_score": 0.87,
  "confidence_level": "high",
  "citations": [
    {
      "source_document": "research_paper.pdf",
      "page": 3,
      "chunk_id": "abc123",
      "excerpt": "This section discusses contextual retrieval, which enriches document chunks...",
      "confidence": 0.92
    }
  ],
  "sources": [
    {
      "chunk_id": "abc123",
      "content": "Chunk text...",
      "score": 0.92,
      "page": 3,
      "method": "hybrid",
      "metadata": {}
    }
  ],
  "query": "What is contextual retrieval?",
  "timestamp": "2026-02-14T10:30:00",
  "latency_ms": 245.5,
  "retrieval_stats": {
    "retrieval_time_ms": 120.3,
    "generation_time_ms": 125.2
  }
}
```

#### POST `/api/v1/compare`
Compare retrieval methods side-by-side.

**Request:**
```json
{
  "q": "What is the Transformer architecture?",
  "k": 5,
  "methods": ["contextual", "bm25", "tfidf"]
}
```

**Response includes:**
- Results from each method (answer, confidence, sources, latency)
- Summary with fastest/highest confidence/recommended method
- Performance comparison (confidence, latency)
- Insights (e.g., "Semantic search outperforms keyword-based for this query")

**Use via Streamlit:**
- Go to "Query System" page → "Compare Methods" tab
- See visual charts comparing all methods

#### GET `/api/v1/health`
Check system health status.

#### GET `/api/v1/metrics`
Get performance metrics.

#### GET `/api/v1/info`
Get system configuration information.

#### GET `/api/v1/audit/logs`
Retrieve audit logs for compliance (supports filtering by user_id, limit, success_only).

#### GET `/api/v1/audit/stats`
Get audit statistics including total queries, success rate, unique users, and average confidence.

##  Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_retrieval.py -v
```

##  Benchmarking

The system includes comprehensive benchmarking that measures:

- **Latency**: Average and P95 query response times
- **Semantic Similarity**: Cosine similarity between generated and ground truth answers
- **Recall@K**: Retrieval effectiveness at K=[1,3,5]

Run benchmarks:
```bash
python scripts/run_benchmarks.py
```

View results:
```bash
cat benchmarks/results.md
```

Expected benchmark output:
```
Method        | Avg Latency (ms) | Semantic Similarity | Recall@5
------------- | ---------------- | ------------------- | --------
hybrid        | 245.2            | 0.8432              | 0.8200
contextual    | 189.3            | 0.8156              | 0.7800
bm25          | 25.4             | 0.7421              | 0.6600
tfidf         | 18.7             | 0.7123              | 0.6200
```

##  Configuration

Edit `.env` file or `config/settings.py` to customize:

### LLM Configuration
```env
LLM_PROVIDER=ollama                    # ollama, openai, anthropic
LLM_MODEL=mistral:latest               # Model name
LLM_BASE_URL=http://localhost:11434   # For Ollama
LLM_TEMPERATURE=0.7
```

### Embedding Configuration
```env
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5  # HuggingFace model
EMBEDDING_DIMENSION=384
```

### Retrieval Configuration
```env
TOP_K=5                                # Number of chunks to retrieve
SIMILARITY_THRESHOLD=0.7
CHUNK_SIZE=512                         # Characters per chunk
CHUNK_OVERLAP=50
```

##  Design Patterns

This implementation showcases enterprise software engineering practices:

### Design Patterns Used

1. **Factory Pattern** - `LLMFactory`, `ChunkingStrategyFactory`
   - Easy creation and switching of components

2. **Strategy Pattern** - `ChunkingStrategy`, `LLMProvider`
   - Interchangeable algorithms for chunking and LLM providers

3. **Facade Pattern** - `QueryEngine`
   - Simplified interface to complex RAG pipeline

4. **Singleton Pattern** - `Settings` (via `@lru_cache`)
   - Single configuration instance

5. **Observer Pattern** - `MetricsCollector`
   - Performance monitoring and metric collection

### SOLID Principles

-  **Single Responsibility**: Each class has one clear purpose
-  **Open/Closed**: Easy to extend (new retrievers, LLM providers) without modification
-  **Liskov Substitution**: Implementations are interchangeable via interfaces
-  **Interface Segregation**: Specific interfaces for different concerns
-  **Dependency Injection**: Components receive dependencies explicitly

##  Project Structure

```
Contextual_RAG/
├── config/                      # Configuration management
│   ├── __init__.py
│   └── settings.py             # Pydantic settings with env support
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── api/                    # API layer
│   │   ├── __init__.py
│   │   └── routes.py           # REST endpoints
│   ├── core/                   # Core business logic
│   │   ├── __init__.py
│   │   ├── llm_factory.py      # LLM provider factory
│   │   ├── document_processor.py  # PDF processing
│   │   ├── chunking_strategies.py # Multiple chunking methods
│   │   ├── retrievers.py       # Retrieval implementations
│   │   └── query_engine.py     # Main orchestrator
│   ├── models/                 # Data models
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic models
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── logger.py           # Logging setup
│       └── metrics.py          # Performance metrics
├── scripts/                    # Utility scripts
│   ├── generate_ground_truth.py
│   └── run_benchmarks.py
├── data/                       # Data directory
│   ├── research_paper.pdf      # Input PDF (you provide)
│   └── ground_truth.json       # Generated QA pairs
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_api.py
├── benchmarks/                 # Benchmark results
│   ├── results.md              # Markdown report
│   └── results.json            # Raw data
├── logs/                       # Application logs
├── chroma_db/                  # Vector store persistence
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
├── .gitignore
├── README.md                   # This file
├── ARCHITECTURE.md             # Architecture documentation
└── BENCHMARK_REPORT.md         # Sample benchmark report
```

##  Key Implementation Details

### Dependency Injection
- **FastAPI routes**: Use `Depends()` for QueryEngine and CacheStore
- **QueryEngine**: Constructor receives LLM and embedding model as explicit parameters
- **Benefits**: Easy testing with mocks, no global state, loose coupling

### LLM Provider Switching
- **Config-based**: Change `llm_provider` in `config/settings.py` or `.env`
- **Factory pattern**: `LLMFactory` abstracts Ollama, OpenAI, etc.
- **Zero code changes**: Just edit config and restart
- **See**: [LLM_SWITCHING.md](LLM_SWITCHING.md) for details

### Retrieval Method Comparison
- **Endpoint**: `POST /api/v1/compare`
- **Compares**: Contextual vs BM25 vs TF-IDF side-by-side
- **Shows**: Confidence, latency, recommended method per query
- **UI**: Available in Streamlit "Compare Methods" tab

### Contextual Retrieval (Anthropic's Approach)

The system implements Anthropic's contextual retrieval by:

1. **Context Generation**: For each chunk, an LLM generates a brief context describing its purpose
2. **Enriched Embedding**: Chunks are embedded with their contextual prefix
3. **Improved Retrieval**: The additional context helps match queries more accurately

Example:
```python
# Original chunk:
"The model achieved 94% accuracy on the test set."

# With context:
"This section presents experimental results. The model achieved 94% accuracy on the test set."

# → Better retrieval for: "What were the experimental results?"
```

### Hybrid Retrieval with RRF

Combines multiple retrieval methods using Reciprocal Rank Fusion (RRF):

```python
score(doc) = Σ_method [ weight(method) / (k + rank_method(doc)) ]
```

This approach leverages:
- **Semantic understanding** from contextual embeddings
- **Keyword matching** from BM25
- **Term frequency signals** from TF-IDF

##  Extending the System

### Adding a New LLM Provider

```python
# In src/core/llm_factory.py

class CustomProvider(LLMProvider):
    def get_llm(self) -> LLM:
        # Return your LLM instance
        pass
    
    def health_check(self) -> bool:
        # Implement health check
        pass

# Register the provider
LLMFactory.register_provider("custom", CustomProvider)
```

### Adding a New Chunking Strategy

```python
# In src/core/chunking_strategies.py

class CustomChunkingStrategy(ChunkingStrategy):
    def chunk_documents(self, documents, **kwargs):
        # Implement your chunking logic
        pass
    
    def get_name(self):
        return "custom"

# Register the strategy
ChunkingStrategyFactory.register_strategy("custom", CustomChunkingStrategy)
```

##  Performance Optimization

For production deployment with hardware constraints:

1. **Use smaller embedding models**: Switch to `BAAI/bge-small-en-v1.5` (384 dim)
2. **Enable caching**: Set `ENABLE_CACHE=true` for repeated queries
3. **Adjust chunk size**: Larger chunks = fewer embeddings = less memory
4. **Disable contextual enrichment**: Set `enable_contextual_retrieval=False` for faster indexing
5. **Use BM25 only**: For lowest latency, use `retrieval_method=bm25`

##  Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Memory Issues
```bash
# Use smaller model
ollama pull mistral:7b-instruct-q4_0

# Update .env
LLM_MODEL=mistral:7b-instruct-q4_0
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

##  License

MIT License - see LICENSE file for details

##  Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

##  Contact

**Author**: Md Marghub Akhtar 
**Role**: Senior AI & Gen AI Engineer 


##  Acknowledgments

- [Anthropic](https://www.anthropic.com/) for the contextual retrieval approach
- [LlamaIndex](https://www.llamaindex.ai/) for RAG framework
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent API framework
- [ChromaDB](https://www.trychroma.com/) for vector storage

---

**Built with attention to production quality, clean architecture, and enterprise standards** 
