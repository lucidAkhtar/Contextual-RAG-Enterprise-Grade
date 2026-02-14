"""
Query engine orchestrating retrieval and generation.
Implements the Facade pattern for simplified RAG pipeline access.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from llama_index.core.schema import TextNode, QueryBundle
from llama_index.core.llms import LLM
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from config.settings import get_settings
from src.core.llm_factory import LLMFactory
from src.core.document_processor import DocumentProcessor
from src.core.chunking_strategies import ChunkingStrategyFactory
from src.core.retrievers import (
    ContextualRetriever,
    BM25Retriever,
    TFIDFRetriever,
    HybridRetriever
)
from src.models.schemas import RetrievalSource
from src.utils.logger import setup_logger
from src.utils.metrics import MetricsCollector

logger = setup_logger(__name__)


class QueryEngine:
    """
    Main query engine orchestrating the RAG pipeline.
    
    Responsibilities:
    - Document ingestion and indexing
    - Multi-method retrieval
    - LLM-based answer generation
    - Performance monitoring
    """
    
    def __init__(
        self,
        pdf_path: Optional[str] = None,
        pdf_paths: Optional[List[str]] = None,
        embed_model: Optional[BaseEmbedding] = None,
        llm: Optional[LLM] = None,
        chunking_strategy: str = "fixed_size",
        enable_contextual_retrieval: bool = True
    ):
        """
        Initialize query engine.
        
        Args:
            pdf_path: Path to single PDF document (deprecated, use pdf_paths)
            pdf_paths: List of paths to PDF documents (for multi-document RAG)
            embed_model: Embedding model instance
            llm: LLM instance
            chunking_strategy: Chunking strategy name
            enable_contextual_retrieval: Enable Anthropic's contextual retrieval
        """
        self.settings = get_settings()
        self.metrics = MetricsCollector()
        
        # Initialize models
        self.embed_model = embed_model or self._init_embedding_model()
        self.llm = llm or LLMFactory.get_llm()
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.chunking_strategy_name = chunking_strategy
        self.enable_contextual_retrieval = enable_contextual_retrieval
        
        # Storage
        self.nodes: List[TextNode] = []
        self.vector_store_index: Optional[VectorStoreIndex] = None
        self.contextual_retriever: Optional[ContextualRetriever] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.tfidf_retriever: Optional[TFIDFRetriever] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.document_names: List[str] = []  # Track loaded documents
        
        # Load and index documents if provided
        if pdf_paths:
            self.ingest_documents(pdf_paths)
        elif pdf_path:
            # Backward compatibility
            self.ingest_document(pdf_path)
        
        logger.info("QueryEngine initialized successfully")
    
    def _init_embedding_model(self) -> BaseEmbedding:
        """Initialize embedding model."""
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        model_name = self.settings.embedding_model
        logger.info(f"Loading embedding model: {model_name}")
        
        return HuggingFaceEmbedding(
            model_name=model_name,
            trust_remote_code=True
        )
    
    def ingest_document(self, pdf_path: str) -> None:
        """
        Ingest and index PDF document.
        
        Args:
            pdf_path: Path to PDF file
        """
        logger.info(f"Ingesting document: {pdf_path}")
        start_time = time.perf_counter()
        
        # Extract document name
        from pathlib import Path
        doc_name = Path(pdf_path).name
        self.document_names.append(doc_name)
        
        # Load PDF
        documents = self.doc_processor.load_pdf(pdf_path)
        
        # Add document source to metadata
        for doc in documents:
            doc.metadata["source_document"] = doc_name
        
        # Chunk documents
        kwargs = {}
        if self.chunking_strategy_name == "semantic":
            kwargs["embed_model"] = self.embed_model
        
        chunking_strategy = ChunkingStrategyFactory.create_strategy(
            self.chunking_strategy_name,
            **kwargs
        )
        self.nodes = chunking_strategy.chunk_documents(documents)
        
        # Initialize vector store (ChromaDB)
        self._init_vector_store()
        
        # Initialize retrievers
        self._init_retrievers()
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"Document ingestion completed in {elapsed:.2f}ms")
        self.metrics.record("ingestion_latency", elapsed)
    
    def ingest_documents(self, pdf_paths: List[str]) -> None:
        """
        Ingest and index multiple PDF documents into a single vector store.
        
        Args:
            pdf_paths: List of paths to PDF files
        """
        logger.info(f"Ingesting {len(pdf_paths)} documents...")
        start_time = time.perf_counter()
        
        all_documents = []
        from pathlib import Path
        
        # Load all PDFs
        for pdf_path in pdf_paths:
            doc_name = Path(pdf_path).name
            self.document_names.append(doc_name)
            logger.info(f"  Loading: {doc_name}")
            
            documents = self.doc_processor.load_pdf(pdf_path)
            
            # Add document source to metadata
            for doc in documents:
                doc.metadata["source_document"] = doc_name
            
            all_documents.extend(documents)
        
        # Chunk all documents together
        kwargs = {}
        if self.chunking_strategy_name == "semantic":
            kwargs["embed_model"] = self.embed_model
        
        chunking_strategy = ChunkingStrategyFactory.create_strategy(
            self.chunking_strategy_name,
            **kwargs
        )
        self.nodes = chunking_strategy.chunk_documents(all_documents)
        
        logger.info(f"Created {len(self.nodes)} chunks from {len(all_documents)} pages")
        
        # Initialize vector store (ChromaDB)
        self._init_vector_store()
        
        # Initialize retrievers
        self._init_retrievers()
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"Multi-document ingestion completed in {elapsed:.2f}ms")
        self.metrics.record("ingestion_latency", elapsed)
    
    def _init_vector_store(self) -> None:
        """Initialize ChromaDB vector store and index."""
        logger.info("Initializing vector store...")
        
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(
            path=self.settings.chroma_persist_dir
        )
        
        # Create or get collection
        collection = chroma_client.get_or_create_collection(
            name="contextual_rag",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=collection)
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index
        self.vector_store_index = VectorStoreIndex(
            nodes=self.nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=True
        )
        
        logger.info("Vector store initialized")
    
    def _init_retrievers(self) -> None:
        """Initialize all retrieval methods."""
        logger.info("Initializing retrievers...")
        
        # Contextual retriever
        self.contextual_retriever = ContextualRetriever(
            nodes=self.nodes,
            embed_model=self.embed_model,
            vector_store_index=self.vector_store_index,
            similarity_top_k=self.settings.top_k,
            use_contextual_enrichment=self.enable_contextual_retrieval
        )
        
        # BM25 retriever
        self.bm25_retriever = BM25Retriever(nodes=self.nodes)
        
        # TF-IDF retriever
        self.tfidf_retriever = TFIDFRetriever(nodes=self.nodes)
        
        # Hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            contextual_retriever=self.contextual_retriever,
            bm25_retriever=self.bm25_retriever,
            tfidf_retriever=self.tfidf_retriever
        )
        
        logger.info("All retrievers initialized")
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        method: str = "hybrid"
    ) -> Tuple[str, List[RetrievalSource], Dict[str, Any]]:
        """
        Execute query and generate answer.
        
        Args:
            query_text: Query string
            top_k: Number of chunks to retrieve
            method: Retrieval method (contextual, bm25, tfidf, hybrid)
        
        Returns:
            Tuple of (answer, sources, stats)
        """
        logger.info(f"Processing query with method '{method}': {query_text[:100]}")
        start_time = time.perf_counter()
        
        # Retrieve relevant chunks
        retrieval_start = time.perf_counter()
        sources = self._retrieve(query_text, top_k, method)
        retrieval_time = (time.perf_counter() - retrieval_start) * 1000
        
        # Generate answer
        generation_start = time.perf_counter()
        answer = self._generate_answer(query_text, sources)
        generation_time = (time.perf_counter() - generation_start) * 1000
        
        # Calculate total time
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Record metrics
        self.metrics.record(f"{method}_retrieval_latency", retrieval_time)
        self.metrics.record(f"{method}_generation_latency", generation_time)
        self.metrics.record(f"{method}_total_latency", total_time)
        
        stats = {
            "retrieval_time_ms": retrieval_time,
            "generation_time_ms": generation_time,
            "total_time_ms": total_time,
            "num_sources": len(sources),
            "method": method
        }
        
        logger.info(f"Query completed in {total_time:.2f}ms")
        
        return answer, sources, stats
    
    def calculate_confidence(self, sources: List[RetrievalSource]) -> Tuple[float, str]:
        """
        Calculate confidence score based on retrieval scores and source quality.
        
        Args:
            sources: List of retrieved sources
            
        Returns:
            Tuple of (confidence_score, confidence_level)
        """
        if not sources:
            return 0.0, "low"
        
        # Calculate confidence based on:
        # 1. Top retrieval scores (normalized 0-1)
        # 2. Score variance (lower variance = higher confidence)
        # 3. Number of sources
        
        scores = [s.score for s in sources]
        
        # Normalize scores to 0-1 range
        max_score = max(scores) if scores else 1.0
        normalized_scores = [s / max_score for s in scores] if max_score > 0 else scores
        
        # Weight: 70% top score, 20% average, 10% consistency
        top_score_weight = normalized_scores[0] if normalized_scores else 0
        avg_score_weight = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0
        
        # Calculate consistency (1 - coefficient of variation)
        if len(normalized_scores) > 1:
            mean_score = sum(normalized_scores) / len(normalized_scores)
            variance = sum((x - mean_score) ** 2 for x in normalized_scores) / len(normalized_scores)
            std_dev = variance ** 0.5
            consistency = 1 - (std_dev / mean_score) if mean_score > 0 else 0
            consistency = max(0, min(1, consistency))  # Clamp to 0-1
        else:
            consistency = 1.0
        
        # Final confidence score
        confidence_score = (
            0.70 * top_score_weight +
            0.20 * avg_score_weight +
            0.10 * consistency
        )
        
        # Determine confidence level
        if confidence_score >= 0.75:
            confidence_level = "high"
        elif confidence_score >= 0.50:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        return round(confidence_score, 3), confidence_level
    
    def create_citations(self, sources: List[RetrievalSource], max_citations: int = 3) -> List[Dict[str, Any]]:
        """
        Create citation objects from retrieval sources.
        
        Args:
            sources: List of retrieved sources
            max_citations: Maximum number of citations to return
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        for source in sources[:max_citations]:
            # Create excerpt (first 200 chars of content)
            excerpt = source.content[:200] + "..." if len(source.content) > 200 else source.content
            
            # Normalize score to 0-1 confidence
            citation_confidence = min(1.0, source.score / max(s.score for s in sources) if sources else 1.0)
            
            citation = {
                "source_document": source.source_document or "unknown",
                "page": source.page,
                "chunk_id": source.chunk_id,
                "excerpt": excerpt,
                "confidence": round(citation_confidence, 3)
            }
            citations.append(citation)
        
        return citations
    
    def _retrieve(
        self,
        query: str,
        top_k: int,
        method: str
    ) -> List[RetrievalSource]:
        """Retrieve relevant chunks using specified method."""
        sources = []
        
        if method == "contextual":
            query_bundle = QueryBundle(query_str=query)
            results = self.contextual_retriever._retrieve(query_bundle)
            
            for node_with_score in results[:top_k]:
                sources.append(RetrievalSource(
                    chunk_id=node_with_score.node.node_id,
                    content=node_with_score.node.metadata.get(
                        "original_text",
                        node_with_score.node.get_content()
                    ),
                    score=node_with_score.score,
                    page=node_with_score.node.metadata.get("page"),
                    source_document=node_with_score.node.metadata.get("source_document"),
                    method="contextual",
                    metadata=node_with_score.node.metadata
                ))
        
        elif method == "bm25":
            results = self.bm25_retriever.retrieve(query, top_k=top_k)
            
            for node, score in results:
                sources.append(RetrievalSource(
                    chunk_id=node.node_id,
                    content=node.get_content(),
                    score=score,
                    page=node.metadata.get("page"),
                    source_document=node.metadata.get("source_document"),
                    method="bm25",
                    metadata=node.metadata
                ))
        
        elif method == "tfidf":
            results = self.tfidf_retriever.retrieve(query, top_k=top_k)
            
            for node, score in results:
                sources.append(RetrievalSource(
                    chunk_id=node.node_id,
                    content=node.get_content(),
                    score=score,
                    page=node.metadata.get("page"),
                    source_document=node.metadata.get("source_document"),
                    method="tfidf",
                    metadata=node.metadata
                ))
        
        elif method == "hybrid":
            results = self.hybrid_retriever.retrieve(query, top_k=top_k)
            
            for node, combined_score, method_scores in results:
                sources.append(RetrievalSource(
                    chunk_id=node.node_id,
                    content=node.get_content(),
                    score=combined_score,
                    page=node.metadata.get("page"),
                    source_document=node.metadata.get("source_document"),
                    method="hybrid",
                    metadata={
                        **node.metadata,
                        "method_scores": method_scores
                    }
                ))
        
        else:
            raise ValueError(f"Unsupported retrieval method: {method}")
        
        return sources
    
    def _generate_answer(
        self,
        query: str,
        sources: List[RetrievalSource]
    ) -> str:
        """Generate answer using LLM and retrieved context."""
        # Prepare context from sources
        context_parts = []
        for idx, source in enumerate(sources, 1):
            context_parts.append(
                f"[Source {idx} - Page {source.page}]\n{source.content}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a helpful assistant answering questions based on provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer the question based ONLY on the provided context
- Be concise and accurate
- If the context doesn't contain enough information, say so
- Cite sources by their numbers when relevant

Answer:"""
        
        # Generate response
        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer."
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return self.metrics.get_all_stats()
