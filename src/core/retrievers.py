"""
Retrieval implementations: Contextual Embeddings, BM25, and TF-IDF.
Implements Contextual Retrieval as proposed by Anthropic.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
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
            self._enrich_nodes_with_context()
        
        logger.info(
            f"Initialized ContextualRetriever with {len(nodes)} nodes, "
            f"contextual_enrichment={use_contextual_enrichment}"
        )
    
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
        
        for node in self.nodes:
            try:
                # Get document-level context
                doc_context = node.metadata.get("document_title", "document")
                page_num = node.metadata.get("page", "unknown")
                
                # Create prompt for context generation
                prompt = f"""Given the following document chunk, provide a brief (1-2 sentences) context describing what this chunk is about and its role in the document.

Document: {doc_context}
Page: {page_num}

Chunk:
{node.get_content()[:500]}...

Context:"""
                
                # Generate context using LLM
                response = llm.complete(prompt)
                contextual_prefix = response.text.strip()
                
                # Store original text and add contextual prefix
                node.metadata["original_text"] = node.get_content()
                node.metadata["contextual_prefix"] = contextual_prefix
                
                # Update node text with context
                enriched_text = f"{contextual_prefix}\n\n{node.get_content()}"
                node.text = enriched_text
                
            except Exception as e:
                logger.warning(f"Error enriching node {node.node_id}: {e}")
                # Keep original text if enrichment fails
                continue
        
        logger.info("Node enrichment completed")
    
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
