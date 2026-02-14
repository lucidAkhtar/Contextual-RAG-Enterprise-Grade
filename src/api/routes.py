"""API routes and endpoints."""

from fastapi import APIRouter, HTTPException, status, Request
from typing import Dict, Any, List
import time
import uuid

from src.models.schemas import (
    QueryRequest,
    QueryResponse,
    RetrievalSource,
    HealthResponse,
    Citation
)
from src.core.query_engine import QueryEngine
from src.core.cache_store import CacheStore
from src.utils.logger import setup_logger
from src import __version__

logger = setup_logger(__name__)

# Create router
router = APIRouter()

# Global instances (initialized in main.py)
query_engine: QueryEngine = None
cache_store: CacheStore = None


def set_query_engine(engine: QueryEngine) -> None:
    """Set the global query engine instance."""
    global query_engine
    query_engine = engine


def set_cache_store(store: CacheStore) -> None:
    """Set the global cache store instance."""
    global cache_store
    cache_store = store


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the RAG system",
    description="""
    Query the Contextual RAG system to retrieve relevant information and generate an answer.
    
    **Retrieval Methods:**
    - `contextual`: Uses Anthropic's contextual retrieval with enriched embeddings
    - `bm25`: Traditional BM25 probabilistic retrieval
    - `tfidf`: TF-IDF based retrieval
    - `hybrid`: Combines all methods using reciprocal rank fusion (recommended)
    
    **Parameters:**
    - `q`: The query string (required)
    - `k`: Number of chunks to retrieve (1-20, default: 5)
    - `retrieval_method`: Method to use (default: hybrid)
    
    **Returns:**
    - Generated answer from LLM
    - Retrieved source chunks with scores
    - Performance metrics
    """,
    response_description="Query response with answer and sources",
    tags=["Query"]
)
async def query(request_body: QueryRequest, request: Request) -> QueryResponse:
    """
    Process a query and return generated answer with sources.
    
    Args:
        request_body: Query request containing query text and parameters
        request: FastAPI request object for audit logging
    
    Returns:
        QueryResponse with answer, confidence, citations, and sources
    
    Raises:
        HTTPException: If query processing fails or engine not initialized
    """
    if query_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query engine not initialized. Please check server logs."
        )
    
    # Generate audit log ID
    log_id = str(uuid.uuid4())
    
    # Extract client info for audit
    client_host = request.client.host if request.client else "unknown"
    
    try:
        logger.info(f"Received query: {request_body.q[:100]}")
        start_time = time.perf_counter()
        cache_hit = False
        
        # Check cache first (if cache store is initialized)
        if cache_store is not None:
            cached_result = cache_store.get_cached_query(
                query=request_body.q,
                method=request_body.retrieval_method,
                top_k=request_body.k
            )
            
            if cached_result is not None:
                cache_hit = True
                answer = cached_result['answer']
                sources = [RetrievalSource(**s) for s in cached_result['sources']]
                stats = cached_result['stats']
                stats['cache_hit'] = True
                
                # Get cached confidence and citations if available
                confidence_score = cached_result.get('confidence_score', 0.85)
                confidence_level = cached_result.get('confidence_level', 'medium')
                citations = [Citation(**c) for c in cached_result.get('citations', [])]
                
                logger.info(f"Cache hit for query: {request_body.q[:50]}...")
            else:
                # Cache miss - execute query
                answer, sources, stats = query_engine.query(
                    query_text=request_body.q,
                    top_k=request_body.k,
                    method=request_body.retrieval_method
                )
                stats['cache_hit'] = False
                
                # Calculate confidence score
                confidence_score, confidence_level = query_engine.calculate_confidence(sources)
                
                # Create citations from top sources
                citations_dict = query_engine.create_citations(sources, max_citations=3)
                citations = [Citation(**c) for c in citations_dict]
                
                # Store in cache with confidence and citations
                cache_store.set_cached_query(
                    query=request_body.q,
                    method=request_body.retrieval_method,
                    top_k=request_body.k,
                    answer=answer,
                    sources=[s.dict() for s in sources],
                    stats={
                        **stats,
                        'confidence_score': confidence_score,
                        'confidence_level': confidence_level,
                        'citations': [c.dict() for c in citations]
                    }
                )
        else:
            # No cache store - execute query directly
            answer, sources, stats = query_engine.query(
                query_text=request_body.q,
                top_k=request_body.k,
                method=request_body.retrieval_method
            )
            stats['cache_hit'] = False
            
            # Calculate confidence and citations
            confidence_score, confidence_level = query_engine.calculate_confidence(sources)
            citations_dict = query_engine.create_citations(sources, max_citations=3)
            citations = [Citation(**c) for c in citations_dict]
        
        # Calculate total latency (including cache lookup)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Get list of documents accessed
        documents_accessed = list(set([s.source_document for s in sources if s.source_document]))
        
        # Store in query history
        if cache_store is not None:
            cache_store.add_query_history(
                query=request_body.q,
                answer=answer,
                method=request_body.retrieval_method,
                top_k=request_body.k,
                sources=[s.dict() for s in sources],
                latency_ms=latency_ms,
                cache_hit=cache_hit
            )
            
            # Add enterprise audit log
            cache_store.add_audit_log(
                log_id=log_id,
                query=request_body.q,
                answer=answer,
                confidence_score=confidence_score,
                documents_accessed=documents_accessed,
                retrieval_method=request_body.retrieval_method,
                latency_ms=latency_ms,
                cache_hit=cache_hit,
                user_id=request_body.user_id,
                session_id=request_body.session_id,
                ip_address=client_host,
                success=True,
                metadata={
                    'confidence_level': confidence_level,
                    'num_citations': len(citations),
                    'num_sources': len(sources)
                }
            )
        
        # Build response with confidence and citations
        response = QueryResponse(
            answer=answer,
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            citations=citations,
            sources=sources,
            query=request_body.q,
            latency_ms=latency_ms,
            retrieval_stats=stats
        )
        
        logger.info(f"Query processed successfully in {latency_ms:.2f}ms (confidence={confidence_score:.2f}, cache_hit={cache_hit})")
        return response
        
    except ValueError as e:
        # Log failed query in audit
        if cache_store is not None:
            cache_store.add_audit_log(
                log_id=log_id,
                query=request_body.q,
                answer="",
                confidence_score=0.0,
                documents_accessed=[],
                retrieval_method=request_body.retrieval_method,
                latency_ms=0.0,
                cache_hit=False,
                user_id=request_body.user_id,
                session_id=request_body.session_id,
                ip_address=client_host,
                success=False,
                error_message=f"Invalid request: {str(e)}"
            )
        
        logger.error(f"Invalid query request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Log failed query in audit
        if cache_store is not None:
            cache_store.add_audit_log(
                log_id=log_id,
                query=request_body.q,
                answer="",
                confidence_score=0.0,
                documents_accessed=[],
                retrieval_method=request_body.retrieval_method,
                latency_ms=0.0,
                cache_hit=False,
                user_id=request_body.user_id,
                session_id=request_body.session_id,
                ip_address=client_host,
                success=False,
                error_message=str(e)
            )
        
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check the health status of the RAG system and its components.",
    tags=["System"]
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with system status
    """
    try:
        llm_status = "unknown"
        vector_store_status = "unknown"
        
        if query_engine is not None:
            llm_status = "healthy"
            vector_store_status = "healthy" if query_engine.vector_store_index else "not_initialized"
            overall_status = "healthy"
        else:
            overall_status = "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            version=__version__,
            llm_status=llm_status,
            vector_store_status=vector_store_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version=__version__,
            llm_status="error",
            vector_store_status="error"
        )


@router.get(
    "/metrics",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get performance metrics",
    description="Retrieve performance metrics collected during query processing.",
    tags=["System"]
)
async def get_metrics() -> Dict[str, Any]:
    """
    Get collected performance metrics.
    
    Returns:
        Dictionary of metrics with statistics
    """
    if query_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query engine not initialized"
        )
    
    try:
        metrics = query_engine.get_metrics()
        return {
            "status": "success",
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/info",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get system information",
    description="Get information about the RAG system configuration.",
    tags=["System"]
)
async def get_info() -> Dict[str, Any]:
    """
    Get system information and configuration.
    
    Returns:
        System information dictionary
    """
    if query_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query engine not initialized"
        )
    
    return {
        "version": __version__,
        "retrieval_methods": ["contextual", "bm25", "tfidf", "hybrid"],
        "chunking_strategy": query_engine.chunking_strategy_name,
        "num_chunks": len(query_engine.nodes),
        "contextual_retrieval_enabled": query_engine.enable_contextual_retrieval,
        "cache_enabled": cache_store is not None,
        "document_names": query_engine.document_names
    }


@router.get(
    "/cache/stats",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get cache statistics",
    description="Get statistics about cache performance including hit rate and total queries.",
    tags=["Cache"]
)
async def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache performance statistics.
    
    Returns:
        Cache statistics dictionary
    """
    if cache_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache store not initialized"
        )
    
    try:
        stats = cache_store.get_cache_statistics()
        return {
            "status": "success",
            "cache_stats": stats
        }
    except Exception as e:
        logger.error(f"Error retrieving cache stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/cache/history",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get query history",
    description="Retrieve recent query history for analytics and monitoring.",
    tags=["Cache"]
)
async def get_query_history(
    limit: int = 50,
    method: str = None
) -> Dict[str, Any]:
    """
    Get query history.
    
    Args:
        limit: Maximum number of entries (default: 50, max: 500)
        method: Filter by retrieval method (optional)
    
    Returns:
        Query history list
    """
    if cache_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache store not initialized"
        )
    
    try:
        # Limit maximum to 500
        limit = min(limit, 500)
        
        history = cache_store.get_query_history(limit=limit, method=method)
        
        return {
            "status": "success",
            "count": len(history),
            "limit": limit,
            "history": history
        }
    except Exception as e:
        logger.error(f"Error retrieving query history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post(
    "/cache/clear",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Clear cache",
    description="Clear all cached query results. History is preserved.",
    tags=["Cache"]
)
async def clear_cache() -> Dict[str, Any]:
    """
    Clear all cached queries.
    
    Returns:
        Success confirmation
    """
    if cache_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache store not initialized"
        )
    
    try:
        cache_store.clear_cache()
        logger.info("Cache cleared via API")
        return {
            "status": "success",
            "message": "Cache cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post(
    "/cache/cleanup",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Cleanup expired cache",
    description="Remove expired cache entries based on TTL.",
    tags=["Cache"]
)
async def cleanup_cache() -> Dict[str, Any]:
    """
    Cleanup expired cache entries.
    
    Returns:
        Number of entries removed
    """
    if cache_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache store not initialized"
        )
    
    try:
        removed = cache_store.cleanup_expired_cache()
        logger.info(f"Cache cleanup completed: {removed} entries removed")
        return {
            "status": "success",
            "entries_removed": removed,
            "message": f"Removed {removed} expired cache entries"
        }
    except Exception as e:
        logger.error(f"Error during cache cleanup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/audit/logs",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get audit logs",
    description="Retrieve audit logs for compliance and security monitoring. Supports filtering by user, date range, and success status.",
    tags=["Audit"]
)
async def get_audit_logs(
    limit: int = 50,
    user_id: str = None,
    success_only: bool = None
) -> Dict[str, Any]:
    """
    Get audit logs with optional filtering.
    
    Args:
        limit: Maximum number of entries (default: 50, max: 500)
        user_id: Filter by user ID (optional)
        success_only: Filter by success status (optional)
    
    Returns:
        Audit logs list with metadata
    """
    if cache_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache store not initialized"
        )
    
    try:
        # Limit maximum to 500
        limit = min(limit, 500)
        
        logs = cache_store.get_audit_logs(
            limit=limit,
            user_id=user_id,
            success_only=success_only
        )
        
        return {
            "status": "success",
            "count": len(logs),
            "limit": limit,
            "audit_logs": logs
        }
    except Exception as e:
        logger.error(f"Error retrieving audit logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/audit/stats",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get audit statistics",
    description="Get aggregated audit statistics for compliance reporting.",
    tags=["Audit"]
)
async def get_audit_stats() -> Dict[str, Any]:
    """
    Get audit statistics for compliance reporting.
    
    Returns:
        Audit statistics including query counts, users, and confidence metrics
    """
    if cache_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache store not initialized"
        )
    
    try:
        stats = cache_store.get_audit_statistics()
        return {
            "status": "success",
            "audit_stats": stats
        }
    except Exception as e:
        logger.error(f"Error retrieving audit stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
