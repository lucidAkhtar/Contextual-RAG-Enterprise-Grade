"""
Retrieval implementations: Contextual Embeddings, BM25, and TF-IDF.
Implements Contextual Retrieval as proposed by Anthropic.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from llama_index.core.schema import TextNode, QueryBundle, NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.embeddings import BaseEmbedding

from src.utils.logger import setup_logger
from src.core.llm_factory import LLMFactory

logger = setup_logger(__name__)


class ContextualRetriever(BaseRetriever):
    """
    Contextual Retrieval implementation following Anthropic's approach.
    
    Key Innovation:
    - Enriches each chunk with contextual information from surrounding document
    - Uses LLM to generate situating context for each chunk
    - Embeds chunks with their context for better retrieval
    
    Reference: https://www.anthropic.com/news/contextual-retrieval
    """
    
    def __init__(
        self,
        nodes: List[TextNode],
        embed_model: BaseEmbedding,
        vector_store_index: VectorStoreIndex,
        similarity_top_k: int = 5,
        use_contextual_enrichment: bool = True
    ):
        super().__init__()
        self.nodes = nodes
        self.embed_model = embed_model
        self.vector_store_index = vector_store_index
        self.similarity_top_k = similarity_top_k
        self.use_contextual_enrichment = use_contextual_enrichment
        
        # Generate contextual embeddings if enabled
        if self.use_contextual_enrichment:
            # Check if nodes are already enriched (should be done in query_engine now)
            already_enriched = any(
                "contextual_prefix" in node.metadata 
                for node in self.nodes[:min(5, len(self.nodes))]  # Check first 5 nodes
            )
            
            if already_enriched:
                logger.info("✓ Nodes already enriched (done before embedding phase)")
            else:
                # Fallback: If not enriched yet, do it now (shouldn't happen)
                logger.warning("Nodes not enriched yet - enriching now (this should happen before embedding!)")
                logger.info("=" * 60)
                logger.info("CONTEXTUAL ENRICHMENT PHASE")
                logger.info("=" * 60)
                if self._nodes_already_enriched():
                    logger.info("✓ Found cached contextual enrichment - skipping LLM calls")
                else:
                    logger.info(f"Starting contextual enrichment for {len(nodes)} chunks...")
                    logger.info("This will make 132 LLM calls to Ollama (takes 2-3 minutes)")
                    self._enrich_nodes_with_context()
                    logger.info("=" * 60)
                    logger.info("✓ CONTEXTUAL ENRICHMENT COMPLETE")
                    logger.info("=" * 60)
        
        logger.info(
            f"Initialized ContextualRetriever with {len(nodes)} nodes, "
            f"contextual_enrichment={use_contextual_enrichment}"
        )
    
    def _get_content_hash(self, text: str) -> str:
        """Generate deterministic hash from chunk content."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def _nodes_already_enriched(self) -> bool:
        """
        Check if nodes already have contextual enrichment cached.
        Loads from disk cache if available.
        Uses content hash instead of node_id for deterministic caching.
        """
        cache_path = Path("chroma_db/contextual_enrichment_cache.json")
        
        if not cache_path.exists():
            logger.info("No contextual enrichment cache found")
            return False
        
        try:
            # Load enrichment cache
            with open(cache_path, 'r', encoding='utf-8') as f:
                enrichment_cache = json.load(f)
            
            # Apply cached enrichments to nodes
            applied_count = 0
            for node in self.nodes:
                # Use content hash as key (deterministic across runs)
                content_hash = self._get_content_hash(node.get_content())
                
                if content_hash in enrichment_cache:
                    node.metadata["contextual_prefix"] = enrichment_cache[content_hash]["contextual_prefix"]
                    node.metadata["original_text"] = enrichment_cache[content_hash]["original_text"]
                    # Restore enriched text
                    enriched_text = f"{node.metadata['contextual_prefix']}\n\n{node.metadata['original_text']}"
                    node.text = enriched_text
                    applied_count += 1
            
            if applied_count >= len(self.nodes) * 0.8:  # 80% threshold
                logger.info(
                    f"✓ Loaded contextual enrichment from cache "
                    f"({applied_count}/{len(self.nodes)} nodes)"
                )
                return True
            else:
                logger.warning(
                    f"Cache incomplete: only {applied_count}/{len(self.nodes)} nodes found - will re-enrich"
                )
                return False
                
        except Exception as e:
            logger.warning(f"Failed to load enrichment cache: {e}")
            return False
    
    def _enrich_nodes_with_context(self) -> None:
        """
        Enrich nodes with contextual information using LLM.
        
        Anthropic's approach:
        1. For each chunk, generate a concise context explaining its purpose
        2. Prepend this context to the chunk before embedding
        3. This helps retrieval by providing additional semantic signals
        """
        logger.info("Enriching nodes with contextual information...")
        
        llm = LLMFactory.get_llm()
        total = len(self.nodes)
        enriched_count = 0
        failed_count = 0
        
        for idx, node in enumerate(self.nodes, 1):
            try:
                # Get document-level context
                doc_context = node.metadata.get("document_title", "document")
                page_num = node.metadata.get("page", "unknown")
                
                # Create improved prompt for distinctive context generation
                prompt = f"""You are analyzing a scientific research paper: "{doc_context}" (Page {page_num}).

Extract and list the MOST IMPORTANT technical concepts, methods, and key information from this text chunk. Focus on:
- Specific model names, architectures, or algorithms mentioned
- Key metrics, results, or performance numbers
- Technical terms and domain-specific vocabulary
- Main findings, contributions, or claims
- Important equations, formulas, or mathematical concepts

Provide a concise summary (2-3 sentences max) that captures the UNIQUE aspects of this chunk using precise technical language. Avoid generic phrases like "this chunk discusses" or "this section describes".

Text chunk:
{node.get_content()[:600]}

Key technical summary:"""
                
                # Generate context using LLM
                response = llm.complete(prompt)
                contextual_prefix = response.text.strip()
                
                # Store original text and add contextual prefix
                node.metadata["original_text"] = node.get_content()
                node.metadata["contextual_prefix"] = contextual_prefix
                
                # Update node text with context
                enriched_text = f"{contextual_prefix}\n\n{node.get_content()}"
                node.text = enriched_text
                enriched_count += 1
                
                # Log progress every 10 chunks
                if idx % 10 == 0 or idx == total:
                    logger.info(f"Progress: {idx}/{total} chunks enriched ({idx/total*100:.1f}%)")
                
            except Exception as e:
                logger.warning(f"Error enriching node {node.node_id}: {e}")
                failed_count += 1
                # Keep original text if enrichment fails
                continue
        
        # Save enrichment cache to disk
        self._save_enrichment_cache()
        
        logger.info(f"Node enrichment completed: {enriched_count} succeeded, {failed_count} failed")
        logger.info("All LLM calls finished - Ollama is now idle")
    
    def _save_enrichment_cache(self) -> None:
        """Save contextual enrichment to disk cache for reuse.
        Uses content hash as key for deterministic caching across runs.
        """
        cache_path = Path("chroma_db/contextual_enrichment_cache.json")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            enrichment_cache = {}
            for node in self.nodes:
                if "contextual_prefix" in node.metadata:
                    # Use content hash as key (deterministic across runs)
                    content_hash = self._get_content_hash(node.metadata.get("original_text", node.get_content()))
                    enrichment_cache[content_hash] = {
                        "contextual_prefix": node.metadata["contextual_prefix"],
                        "original_text": node.metadata.get("original_text", "")
                    }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(enrichment_cache, f, indent=2)
            
            logger.info(f"✓ Saved contextual enrichment cache ({len(enrichment_cache)} nodes)")
            
        except Exception as e:
            logger.warning(f"Failed to save enrichment cache: {e}")
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes using contextual embeddings.
        
        Args:
            query_bundle: Query with optional filters
        
        Returns:
            List of nodes with relevance scores
        """
        # Get query embedding
        query_embedding = self.embed_model.get_query_embedding(query_bundle.query_str)
        
        # Query vector store
        query_obj = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self.similarity_top_k
        )
        
        query_result = self.vector_store_index.vector_store.query(query_obj)
        
        # Convert to NodeWithScore format
        nodes_with_scores = []
        for node_id, similarity in zip(query_result.ids, query_result.similarities):
            # Find corresponding node
            node = next((n for n in self.nodes if n.node_id == node_id), None)
            if node:
                nodes_with_scores.append(
                    NodeWithScore(node=node, score=similarity)
                )
        
        logger.info(
            f"ContextualRetriever retrieved {len(nodes_with_scores)} nodes "
            f"for query: '{query_bundle.query_str[:50]}...'"
        )
        
        return nodes_with_scores


class BM25Retriever:
    """
    BM25 (Best Matching 25) retrieval.
    Traditional probabilistic information retrieval method.
    """
    
    def __init__(self, nodes: List[TextNode]):
        self.nodes = nodes
        self.node_texts = [node.get_content() for node in nodes]
        
        # Tokenize for BM25
        tokenized_corpus = [text.lower().split() for text in self.node_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info(f"Initialized BM25Retriever with {len(nodes)} nodes")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[TextNode, float]]:
        """
        Retrieve using BM25 scoring.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of (node, score) tuples
        """
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = [
            (self.nodes[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0  # Only include non-zero scores
        ]
        
        logger.info(
            f"BM25Retriever retrieved {len(results)} nodes "
            f"for query: '{query[:50]}...'"
        )
        
        return results


class TFIDFRetriever:
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) retrieval.
    Classic information retrieval baseline.
    """
    
    def __init__(self, nodes: List[TextNode]):
        self.nodes = nodes
        self.node_texts = [node.get_content() for node in nodes]
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Fit and transform corpus
        self.tfidf_matrix = self.vectorizer.fit_transform(self.node_texts)
        
        logger.info(f"Initialized TFIDFRetriever with {len(nodes)} nodes")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[TextNode, float]]:
        """
        Retrieve using TF-IDF cosine similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of (node, score) tuples
        """
        # Transform query
        query_vec = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            (self.nodes[idx], float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > 0
        ]
        
        logger.info(
            f"TFIDFRetriever retrieved {len(results)} nodes "
            f"for query: '{query[:50]}...'"
        )
        
        return results


class HybridRetriever:
    """
    Hybrid retrieval combining multiple methods.
    Uses weighted scoring or rank fusion.
    """
    
    def __init__(
        self,
        contextual_retriever: Optional[ContextualRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        tfidf_retriever: Optional[TFIDFRetriever] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        self.contextual_retriever = contextual_retriever
        self.bm25_retriever = bm25_retriever
        self.tfidf_retriever = tfidf_retriever
        
        # Default weights
        self.weights = weights or {
            "contextual": 0.5,
            "bm25": 0.3,
            "tfidf": 0.2
        }
        
        logger.info(f"Initialized HybridRetriever with weights: {self.weights}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[TextNode, float, Dict[str, float]]]:
        """
        Retrieve using hybrid approach with reciprocal rank fusion.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of (node, combined_score, method_scores) tuples
        """
        all_results: Dict[str, List] = {}
        
        # Get results from each retriever
        if self.contextual_retriever:
            query_bundle = QueryBundle(query_str=query)
            contextual_results = self.contextual_retriever._retrieve(query_bundle)
            all_results["contextual"] = [
                (ns.node, ns.score) for ns in contextual_results
            ]
        
        if self.bm25_retriever:
            all_results["bm25"] = self.bm25_retriever.retrieve(query, top_k=top_k)
        
        if self.tfidf_retriever:
            all_results["tfidf"] = self.tfidf_retriever.retrieve(query, top_k=top_k)
        
        # Reciprocal Rank Fusion (RRF)
        node_scores: Dict[str, Dict[str, float]] = {}
        k = 60  # RRF constant
        
        for method, results in all_results.items():
            weight = self.weights.get(method, 1.0)
            for rank, (node, score) in enumerate(results, 1):
                node_id = node.node_id
                if node_id not in node_scores:
                    node_scores[node_id] = {
                        "node": node,
                        "scores": {},
                        "combined": 0.0
                    }
                
                rrf_score = weight * (1.0 / (k + rank))
                node_scores[node_id]["scores"][method] = score
                node_scores[node_id]["combined"] += rrf_score
        
        # Sort by combined score
        sorted_results = sorted(
            node_scores.values(),
            key=lambda x: x["combined"],
            reverse=True
        )[:top_k]
        
        results = [
            (item["node"], item["combined"], item["scores"])
            for item in sorted_results
        ]
        
        logger.info(
            f"HybridRetriever retrieved {len(results)} nodes "
            f"for query: '{query[:50]}...'"
        )
        
        return results
