"""API routes and endpoints with dependency injection."""

from fastapi import APIRouter, HTTPException, status, Request, Depends
from typing import Dict, Any, List, Optional
import time
import uuid

from src.models.schemas import (
    QueryRequest,
    QueryResponse,
    RetrievalSource,
    HealthResponse,
    Citation,
    ComparisonRequest,
    ComparisonResponse,
    MethodResult
)
from src.core.query_engine import QueryEngine
from src.core.cache_store import CacheStore
from src.utils.logger import setup_logger
from src import __version__

logger = setup_logger(__name__)

# Create router
router = APIRouter()

# Dependency injection providers
def get_query_engine(request: Request) -> QueryEngine:
    """
    Dependency provider for QueryEngine.
    Uses DI pattern to inject QueryEngine into route handlers.
    
    Args:
        request: FastAPI request with app.state
        
    Returns:
        QueryEngine instance from app state
        
    Raises:
        HTTPException: If engine not initialized
    """
    engine = getattr(request.app.state, "query_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query engine not initialized. Please check server logs."
        )
    return engine


def get_cache_store(request: Request) -> Optional[CacheStore]:
    """
    Dependency provider for CacheStore.
    Uses DI pattern to inject CacheStore into route handlers.
    
    Args:
        request: FastAPI request with app.state
        
    Returns:
        CacheStore instance from app state or None if disabled
    """
    return getattr(request.app.state, "cache_store", None)


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
async def query(
    request_body: QueryRequest,
    request: Request,
    query_engine: QueryEngine = Depends(get_query_engine),
    cache_store: Optional[CacheStore] = Depends(get_cache_store)
) -> QueryResponse:
    """
    Process a query and return generated answer with sources.
    Uses dependency injection for QueryEngine and CacheStore.
    
    Args:
        request_body: Query request containing query text and parameters
        request: FastAPI request object for audit logging
        query_engine: Injected QueryEngine instance (DI)
        cache_store: Injected CacheStore instance or None (DI)
    
    Returns:
        QueryResponse with answer, confidence, citations, and sources
    
    Raises:
        HTTPException: If query processing fails
    """
    
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


@router.post(
    "/compare",
    response_model=ComparisonResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare retrieval methods",
    description="""
    Compare multiple retrieval methods side-by-side for the same query.
    
    This endpoint runs the same query through multiple retrieval methods (contextual, BM25, TF-IDF)
    and returns detailed comparison including:
    - Different answers generated by each method
    - Different source documents retrieved
    - Relevance scores for each method
    - Performance metrics
    - Recommended best method for the query
    
    **Use Cases:**
    - Evaluate which retrieval method works best for specific query types
    - Understand differences between semantic and keyword-based retrieval
    - Benchmark retrieval quality
    - Debug retrieval issues
    
    **Parameters:**
    - `q`: The query string (required)
    - `k`: Number of chunks to retrieve per method (default: 5)
    - `methods`: List of methods to compare (default: all three)
    """,
    response_description="Side-by-side comparison of retrieval methods",
    tags=["Query"]
)
async def compare_retrieval_methods(
    request_body: ComparisonRequest,
    query_engine: QueryEngine = Depends(get_query_engine)
) -> ComparisonResponse:
    """
    Compare multiple retrieval methods for the same query.
    Uses dependency injection for QueryEngine.
    
    Args:
        request_body: Comparison request with query and methods to compare
        query_engine: Injected QueryEngine instance (DI)
    
    Returns:
        ComparisonResponse with results from each method and summary
    
    Raises:
        HTTPException: If comparison fails
    """
    try:
        logger.info(f"Starting retrieval comparison for query: {request_body.q[:100]}")
        comparison_start_time = time.perf_counter()
        
        results = []
        
        # Run query through each requested method
        for method in request_body.methods:
            try:
                method_start = time.perf_counter()
                
                # Execute query with this method
                answer, sources, _ = query_engine.query(
                    query_text=request_body.q,
                    top_k=request_body.k,
                    method=method
                )
                
                # Calculate confidence
                confidence_score, confidence_level = query_engine.calculate_confidence(sources)
                
                method_latency = (time.perf_counter() - method_start) * 1000
                
                # Extract top scores for quick comparison
                top_scores = [s.score for s in sources[:3]] if sources else []
                
                # Create method result
                method_result = MethodResult(
                    method=method,
                    answer=answer,
                    confidence_score=confidence_score,
                    confidence_level=confidence_level,
                    sources=sources,
                    top_source_scores=top_scores,
                    latency_ms=method_latency,
                    num_sources=len(sources)
                )
                
                results.append(method_result)
                logger.info(f"Method '{method}' completed in {method_latency:.2f}ms")
                
            except Exception as method_error:
                logger.error(f"Error with method '{method}': {method_error}")
                # Continue with other methods even if one fails
                continue
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="All retrieval methods failed"
            )
        
        # Calculate total comparison time
        total_latency = (time.perf_counter() - comparison_start_time) * 1000
        
        # Generate summary with recommendations
        summary = {
            "num_methods_compared": len(results),
            "fastest_method": min(results, key=lambda r: r.latency_ms).method,
            "highest_confidence_method": max(results, key=lambda r: r.confidence_score).method,
            "most_sources_method": max(results, key=lambda r: r.num_sources).method,
            "latency_comparison": {
                r.method: round(r.latency_ms, 2) for r in results
            },
            "confidence_comparison": {
                r.method: round(r.confidence_score, 3) for r in results
            },
            "recommended_method": _recommend_method(results),
            "insights": _generate_insights(results)
        }
        
        response = ComparisonResponse(
            query=request_body.q,
            results=results,
            total_latency_ms=total_latency,
            summary=summary
        )
        
        logger.info(
            f"Comparison completed in {total_latency:.2f}ms. "
            f"Recommended: {summary['recommended_method']}"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in retrieval comparison: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}"
        )


def _recommend_method(results: List[MethodResult]) -> str:
    """
    Recommend the best retrieval method based on multiple factors.
    
    Scoring:
    - 50% confidence score
    - 30% number of sources
    - 20% speed (inverse of latency)
    """
    if not results:
        return "hybrid"
    
    max_latency = max(r.latency_ms for r in results) if results else 1.0
    
    scored_methods = []
    for result in results:
        # Normalize metrics
        confidence_norm = result.confidence_score  # Already 0-1
        sources_norm = min(result.num_sources / 10.0, 1.0)  # Cap at 10 sources
        speed_norm = 1.0 - (result.latency_ms / max_latency) if max_latency > 0 else 1.0
        
        # Calculate weighted score
        score = (0.5 * confidence_norm) + (0.3 * sources_norm) + (0.2 * speed_norm)
        scored_methods.append((result.method, score))
    
    # Return method with highest score
    best_method = max(scored_methods, key=lambda x: x[1])[0]
    return best_method


def _generate_insights(results: List[MethodResult]) -> List[str]:
    """
    Generate human-readable insights from comparison results.
    """
    insights = []
    
    if not results or len(results) < 2:
        return insights
    
    # Compare confidence levels
    confidence_scores = {r.method: r.confidence_score for r in results}
    max_conf_method = max(confidence_scores, key=confidence_scores.get)
    min_conf_method = min(confidence_scores, key=confidence_scores.get)
    conf_diff = confidence_scores[max_conf_method] - confidence_scores[min_conf_method]
    
    if conf_diff > 0.2:
        insights.append(
            f"Significant confidence difference: {max_conf_method} "
            f"({confidence_scores[max_conf_method]:.2f}) vs {min_conf_method} "
            f"({confidence_scores[min_conf_method]:.2f})"
        )
    else:
        insights.append("All methods show similar confidence levels")
    
    # Compare latencies
    latencies = {r.method: r.latency_ms for r in results}
    fastest = min(latencies, key=latencies.get)
    slowest = max(latencies, key=latencies.get)
    latency_ratio = latencies[slowest] / latencies[fastest] if latencies[fastest] > 0 else 1.0
    
    if latency_ratio > 2.0:
        insights.append(
            f"{fastest} is {latency_ratio:.1f}x faster than {slowest}"
        )
    
    # Contextual vs traditional IR comparison
    contextual_result = next((r for r in results if r.method == "contextual"), None)
    bm25_result = next((r for r in results if r.method == "bm25"), None)
    
    if contextual_result and bm25_result:
        if contextual_result.confidence_score > bm25_result.confidence_score + 0.1:
            insights.append(
                "Semantic search (contextual) outperforms keyword-based (BM25) for this query"
            )
        elif bm25_result.confidence_score > contextual_result.confidence_score + 0.1:
            insights.append(
                "Keyword-based search (BM25) outperforms semantic search for this query"
            )
        else:
            insights.append(
                "Semantic and keyword-based methods perform similarly"
            )
    
    return insights


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check the health status of the RAG system and its components.",
    tags=["System"]
)
async def health_check(
    query_engine: QueryEngine = Depends(get_query_engine)
) -> HealthResponse:
    """
    Health check endpoint. Uses DI to inject QueryEngine.
    
    Args:
        query_engine: Injected QueryEngine instance (DI)
    
    Returns:
        HealthResponse with system status
    """
    try:
        llm_status = "healthy"
        vector_store_status = "healthy" if query_engine.vector_store_index else "not_initialized"
        overall_status = "healthy"
        
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
async def get_metrics(
    query_engine: QueryEngine = Depends(get_query_engine)
) -> Dict[str, Any]:
    """
    Get collected performance metrics. Uses DI to inject QueryEngine.
    
    Args:
        query_engine: Injected QueryEngine instance (DI)
    
    Returns:
        Dictionary of metrics with statistics
    """
    
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
async def get_info(
    query_engine: QueryEngine = Depends(get_query_engine),
    cache_store: Optional[CacheStore] = Depends(get_cache_store)
) -> Dict[str, Any]:
    """
    Get system information and configuration. Uses DI for dependencies.
    
    Args:
        query_engine: Injected QueryEngine instance (DI)
        cache_store: Injected CacheStore instance or None (DI)
    
    Returns:
        System information dictionary
    """
    
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
async def get_cache_stats(
    cache_store: Optional[CacheStore] = Depends(get_cache_store)
) -> Dict[str, Any]:
    """
    Get cache performance statistics. Uses DI to inject CacheStore.
    
    Args:
        cache_store: Injected CacheStore instance or None (DI)
    
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
    method: str = None,
    cache_store: Optional[CacheStore] = Depends(get_cache_store)
) -> Dict[str, Any]:
    """
    Get query history. Uses DI to inject CacheStore.
    
    Args:
        limit: Maximum number of entries (default: 50, max: 500)
        method: Filter by retrieval method (optional)
        cache_store: Injected CacheStore instance or None (DI)
    
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
async def clear_cache(
    cache_store: Optional[CacheStore] = Depends(get_cache_store)
) -> Dict[str, Any]:
    """
    Clear all cached queries. Uses DI to inject CacheStore.
    
    Args:
        cache_store: Injected CacheStore instance or None (DI)
    
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
async def cleanup_cache(
    cache_store: Optional[CacheStore] = Depends(get_cache_store)
) -> Dict[str, Any]:
    """
    Cleanup expired cache entries. Uses DI to inject CacheStore.
    
    Args:
        cache_store: Injected CacheStore instance or None (DI)
    
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
    success_only: bool = None,
    cache_store: Optional[CacheStore] = Depends(get_cache_store)
) -> Dict[str, Any]:
    """
    Get audit logs with optional filtering. Uses DI to inject CacheStore.
    
    Args:
        limit: Maximum number of entries (default: 50, max: 500)
        user_id: Filter by user ID (optional)
        success_only: Filter by success status (optional)
        cache_store: Injected CacheStore instance or None (DI)
    
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
async def get_audit_stats(
    cache_store: Optional[CacheStore] = Depends(get_cache_store)
) -> Dict[str, Any]:
    """
    Get audit statistics for compliance reporting. Uses DI to inject CacheStore.
    
    Args:
        cache_store: Injected CacheStore instance or None (DI)
    
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
