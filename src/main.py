"""
FastAPI application entry point with dependency injection.
Implements enterprise-grade REST API with OpenAPI specification.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from config.settings import get_settings
from src.api.routes import router
from src.core.query_engine import QueryEngine
from src.core.cache_store import CacheStore
from src.utils.logger import setup_logger
from src import __version__, __author__

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Implements dependency injection by storing instances in app.state.
    
    Args:
        app: FastAPI application instance with state for DI
    """
    # Startup
    logger.info("Starting Contextual RAG API...")
    cache_store_instance = None

    try:
        settings = get_settings()
        
        # Initialize cache store
        if settings.enable_cache:
            logger.info("Initializing cache store...")
            cache_store_instance = CacheStore()
            app.state.cache_store = cache_store_instance  # DI: Store in app.state
            logger.info("Cache store initialized successfully")
        else:
            app.state.cache_store = None
            logger.info("Cache disabled in settings")
        
        # Initialize query engine
        pdf_paths = settings.get_pdf_paths_list()
        logger.info(f"Loading {len(pdf_paths)} PDF(s): {', '.join([Path(p).name for p in pdf_paths])}")
        
        # Check if PDFs exist
        missing_pdfs = [p for p in pdf_paths if not Path(p).exists()]
        if missing_pdfs:
            logger.error(f"PDF file(s) not found: {', '.join(missing_pdfs)}")
            logger.warning("Query engine will not be initialized")
            app.state.query_engine = None
        else:
            engine = QueryEngine.create(
                pdf_paths=pdf_paths,
                chunking_strategy="fixed_size",
                enable_contextual_retrieval=True
            )
            app.state.query_engine = engine  # DI: Store in app.state
            logger.info("Query engine initialized successfully")
        
        logger.info("API startup complete")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        logger.warning("API will start but queries will fail")
        app.state.query_engine = None
        app.state.cache_store = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down Contextual RAG API...")
    
    # Close cache store connection
    if cache_store_instance is not None:
        cache_store_instance.close()
        logger.info("Cache store closed")


# Create FastAPI application
app = FastAPI(
    title="Contextual RAG API",
    description="""
    # Contextual Retrieval-Augmented Generation API
    
    Enterprise-grade RAG system implementing Anthropic's contextual retrieval approach.
    
    ## Features
    
    - **Contextual Retrieval**: Enriches chunks with contextual information for better retrieval
    - **Multiple Retrieval Methods**: Supports contextual embeddings, BM25, TF-IDF, and hybrid fusion
    - **Flexible LLM Backend**: Easy switching between Ollama, OpenAI, and other providers
    - **Production-Ready**: Built with FastAPI, comprehensive logging, and metrics
    
    ## Retrieval Methods
    
    - **contextual**: Uses LLM-enriched chunks with semantic embeddings
    - **bm25**: Traditional probabilistic retrieval (Best Matching 25)
    - **tfidf**: Term Frequency-Inverse Document Frequency retrieval
    - **hybrid**: Combines all methods using reciprocal rank fusion (recommended)
    
    ## Usage
    
    1. Check system health: `GET /health`
    2. Query the system: `POST /query`
    3. View metrics: `GET /metrics`
    
    ## Architecture
    
    Built with:
    - LlamaIndex for RAG orchestration
    - ChromaDB for vector storage
    - FastAPI for REST API
    - Ollama/OpenAI for LLM inference
    
    ## Author
    
    {author}
    """.format(author=__author__),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


# Include routers
app.include_router(router, prefix="/api/v1")


# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema with enhanced documentation."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Contextual RAG API",
        version=__version__,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom schema information
    openapi_schema["info"]["x-logo"] = {
        "url": "https://www.anthropic.com/images/icons/menu/menuCorporate.svg"
    }
    
    openapi_schema["info"]["contact"] = {
        "name": __author__,
        "email": "marghu.bakhtar@example.com"
    }
    
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8000",
            "description": "Local development server"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Contextual RAG API",
        "version": __version__,
        "author": __author__,
        "docs": "/docs",
        "health": "/api/v1/health",
        "query_endpoint": "/api/v1/query"
    }


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )
