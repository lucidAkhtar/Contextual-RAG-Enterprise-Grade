"""
Cache and storage layer using TinyDB.
Mimics production-grade database implementation for query caching and history.
"""

import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

from tinydb import TinyDB, Query
from tinydb.table import Document

from config.settings import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CacheStore:
    """
    Production-grade cache and storage layer using TinyDB.
    In production, this would be replaced with Redis.
    
    Features:
    - Query result caching with TTL
    - Query history tracking
    - Performance metrics storage
    - Automatic cache expiration
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize cache store.
        
        Args:
            db_path: Path to TinyDB JSON file (default: ./cache_db/cache.json)
        """
        self.settings = get_settings()
        
        # Setup database path
        if db_path is None:
            db_path = Path("./cache_db/cache.json")
        else:
            db_path = Path(db_path)
        
        # Create directory if needed
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize TinyDB
        self.db = TinyDB(db_path)
        
        # Create tables (similar to MongoDB collections)
        self.cache_table = self.db.table('query_cache')
        self.history_table = self.db.table('query_history')
        self.metrics_table = self.db.table('metrics')
        self.audit_table = self.db.table('audit_logs')  # Enterprise audit logging
        
        logger.info(f"CacheStore initialized at: {db_path}")
    
    def _generate_cache_key(self, query: str, method: str, top_k: int) -> str:
        """
        Generate deterministic cache key from query parameters.
        
        Args:
            query: Query text
            method: Retrieval method
            top_k: Number of results
            
        Returns:
            MD5 hash as cache key
        """
        key_string = f"{query.lower().strip()}|{method}|{top_k}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cached_query(
        self,
        query: str,
        method: str,
        top_k: int
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached query result if available and not expired.
        
        Args:
            query: Query text
            method: Retrieval method
            top_k: Number of results
            
        Returns:
            Cached result or None if not found/expired
        """
        if not self.settings.enable_cache:
            return None
        
        cache_key = self._generate_cache_key(query, method, top_k)
        Q = Query()
        
        try:
            result = self.cache_table.get(Q.cache_key == cache_key)
            
            if result is None:
                logger.debug(f"Cache miss for query: {query[:50]}...")
                return None
            
            # Check TTL expiration
            cached_at = datetime.fromisoformat(result['cached_at'])
            ttl_seconds = self.settings.cache_ttl
            expires_at = cached_at + timedelta(seconds=ttl_seconds)
            
            if datetime.now() > expires_at:
                # Cache expired, remove it
                logger.debug(f"Cache expired for query: {query[:50]}...")
                self.cache_table.remove(Q.cache_key == cache_key)
                return None
            
            logger.info(f"Cache hit for query: {query[:50]}...")
            return result['data']
            
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None
    
    def set_cached_query(
        self,
        query: str,
        method: str,
        top_k: int,
        answer: str,
        sources: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ) -> None:
        """
        Store query result in cache.
        
        Args:
            query: Query text
            method: Retrieval method
            top_k: Number of results
            answer: Generated answer
            sources: Retrieved sources
            stats: Performance statistics
        """
        if not self.settings.enable_cache:
            return
        
        cache_key = self._generate_cache_key(query, method, top_k)
        Q = Query()
        
        try:
            cache_entry = {
                'cache_key': cache_key,
                'query': query,
                'method': method,
                'top_k': top_k,
                'data': {
                    'answer': answer,
                    'sources': sources,
                    'stats': stats
                },
                'cached_at': datetime.now().isoformat(),
                'ttl_seconds': self.settings.cache_ttl
            }
            
            # Upsert (update if exists, insert if not)
            self.cache_table.upsert(
                Document(cache_entry, doc_id=cache_key.__hash__()),
                Q.cache_key == cache_key
            )
            
            logger.debug(f"Cached query result: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error caching query: {e}")
    
    def add_query_history(
        self,
        query: str,
        answer: str,
        method: str,
        top_k: int,
        sources: List[Dict[str, Any]],
        latency_ms: float,
        cache_hit: bool = False
    ) -> None:
        """
        Store query in history for analytics.
        
        Args:
            query: Query text
            answer: Generated answer
            method: Retrieval method
            top_k: Number of results
            sources: Retrieved sources
            latency_ms: Query latency
            cache_hit: Whether result came from cache
        """
        try:
            history_entry = {
                'query': query,
                'answer': answer,
                'method': method,
                'top_k': top_k,
                'num_sources': len(sources),
                'source_pages': [s.get('page') for s in sources if s.get('page')],
                'latency_ms': latency_ms,
                'cache_hit': cache_hit,
                'timestamp': datetime.now().isoformat()
            }
            
            self.history_table.insert(history_entry)
            logger.debug(f"Added query to history: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error adding to history: {e}")
    
    def get_query_history(
        self,
        limit: int = 100,
        method: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve query history for analytics.
        
        Args:
            limit: Maximum number of entries to return
            method: Filter by retrieval method (optional)
            
        Returns:
            List of query history entries
        """
        try:
            Q = Query()
            
            if method:
                results = self.history_table.search(Q.method == method)
            else:
                results = self.history_table.all()
            
            # Sort by timestamp descending and limit
            results = sorted(
                results,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )[:limit]
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving history: {e}")
            return []
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache stats
        """
        try:
            total_queries = len(self.history_table)
            
            if total_queries == 0:
                return {
                    'total_queries': 0,
                    'cache_hits': 0,
                    'cache_misses': 0,
                    'hit_rate': 0.0,
                    'cached_entries': 0
                }
            
            Q = Query()
            cache_hits = len(self.history_table.search(Q.cache_hit == True))
            cache_misses = total_queries - cache_hits
            hit_rate = (cache_hits / total_queries) * 100 if total_queries > 0 else 0
            
            return {
                'total_queries': total_queries,
                'cache_hits': cache_hits,
                'cache_misses': cache_misses,
                'hit_rate': round(hit_rate, 2),
                'cached_entries': len(self.cache_table)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def clear_cache(self) -> None:
        """Clear all cached queries."""
        try:
            self.cache_table.truncate()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def clear_history(self) -> None:
        """Clear query history."""
        try:
            self.history_table.truncate()
            logger.info("History cleared")
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
    
    def cleanup_expired_cache(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        try:
            Q = Query()
            all_entries = self.cache_table.all()
            removed = 0
            
            for entry in all_entries:
                cached_at = datetime.fromisoformat(entry['cached_at'])
                ttl = entry.get('ttl_seconds', self.settings.cache_ttl)
                expires_at = cached_at + timedelta(seconds=ttl)
                
                if datetime.now() > expires_at:
                    self.cache_table.remove(Q.cache_key == entry['cache_key'])
                    removed += 1
            
            if removed > 0:
                logger.info(f"Cleaned up {removed} expired cache entries")
            
            return removed
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            return 0
    
    def add_audit_log(
        self,
        log_id: str,
        query: str,
        answer: str,
        confidence_score: float,
        documents_accessed: List[str],
        retrieval_method: str,
        latency_ms: float,
        cache_hit: bool,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add audit log entry for compliance and security tracking.
        
        Args:
            log_id: Unique audit log identifier
            query: User query
            answer: Generated answer
            confidence_score: Confidence score (0-1)
            documents_accessed: List of documents accessed
            retrieval_method: Retrieval method used
            latency_ms: Query latency
            cache_hit: Whether result from cache
            user_id: User identifier
            session_id: Session identifier
            ip_address: Client IP address
            success: Whether query succeeded
            error_message: Error message if failed
            metadata: Additional metadata
        """
        try:
            audit_entry = {
                'log_id': log_id,
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'session_id': session_id,
                'ip_address': ip_address,
                'query': query,
                'answer': answer[:500],  # Truncate for storage efficiency
                'confidence_score': confidence_score,
                'documents_accessed': documents_accessed,
                'retrieval_method': retrieval_method,
                'latency_ms': latency_ms,
                'cache_hit': cache_hit,
                'success': success,
                'error_message': error_message,
                'metadata': metadata or {}
            }
            
            self.audit_table.insert(audit_entry)
            logger.debug(f"Audit log created: {log_id}")
            
        except Exception as e:
            logger.error(f"Error adding audit log: {e}")
    
    def get_audit_logs(
        self,
        limit: int = 100,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        success_only: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit logs with optional filtering.
        
        Args:
            limit: Maximum number of entries
            user_id: Filter by user ID
            start_date: Filter by start date
            end_date: Filter by end date
            success_only: Filter by success status
            
        Returns:
            List of audit log entries
        """
        try:
            Q = Query()
            results = self.audit_table.all()
            
            # Apply filters
            if user_id:
                results = [r for r in results if r.get('user_id') == user_id]
            
            if success_only is not None:
                results = [r for r in results if r.get('success') == success_only]
            
            if start_date:
                start_iso = start_date.isoformat()
                results = [r for r in results if r.get('timestamp', '') >= start_iso]
            
            if end_date:
                end_iso = end_date.isoformat()
                results = [r for r in results if r.get('timestamp', '') <= end_iso]
            
            # Sort by timestamp descending and limit
            results = sorted(
                results,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )[:limit]
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving audit logs: {e}")
            return []
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """
        Get audit statistics for compliance reporting.
        
        Returns:
            Dictionary with audit stats
        """
        try:
            total_logs = len(self.audit_table)
            
            if total_logs == 0:
                return {
                    'total_queries': 0,
                    'successful_queries': 0,
                    'failed_queries': 0,
                    'unique_users': 0,
                    'unique_documents': 0,
                    'avg_confidence': 0.0
                }
            
            Q = Query()
            all_logs = self.audit_table.all()
            
            successful = len([l for l in all_logs if l.get('success', True)])
            failed = total_logs - successful
            
            unique_users = len(set([l.get('user_id') for l in all_logs if l.get('user_id')]))
            
            all_docs = []
            for log in all_logs:
                all_docs.extend(log.get('documents_accessed', []))
            unique_documents = len(set(all_docs))
            
            confidences = [l.get('confidence_score', 0.0) for l in all_logs if l.get('confidence_score') is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                'total_queries': total_logs,
                'successful_queries': successful,
                'failed_queries': failed,
                'unique_users': unique_users if unique_users > 0 else 1,
                'unique_documents': unique_documents,
                'avg_confidence': round(avg_confidence, 3)
            }
            
        except Exception as e:
            logger.error(f"Error getting audit stats: {e}")
            return {}
    
    def close(self) -> None:
        """Close database connection."""
        try:
            self.db.close()
            logger.info("CacheStore connection closed")
        except Exception as e:
            logger.error(f"Error closing CacheStore: {e}")
