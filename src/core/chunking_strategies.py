"""
Chunking strategies for document processing.
Implements Strategy pattern for different chunking approaches.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from llama_index.core.schema import Document, TextNode
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser
)
from config.settings import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[TextNode]:
        """
        Chunk documents into nodes.
        
        Args:
            documents: List of documents to chunk
            **kwargs: Strategy-specific parameters
        
        Returns:
            List of text nodes
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name."""
        pass


class FixedSizeChunkingStrategy(ChunkingStrategy):
    """
    Fixed-size chunking with configurable overlap.
    Traditional approach suitable for most use cases.
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separator: str = " "
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separator = separator
        
        self.splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=self.separator
        )
        
        logger.info(
            f"Initialized FixedSizeChunking: size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )
    
    def chunk_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[TextNode]:
        """Chunk documents using fixed size."""
        nodes = self.splitter.get_nodes_from_documents(documents)
        logger.info(f"Created {len(nodes)} chunks using FixedSizeChunking")
        return nodes
    
    def get_name(self) -> str:
        return "fixed_size"


class SemanticChunkingStrategy(ChunkingStrategy):
    """
    Semantic chunking based on embedding similarity.
    Groups semantically similar sentences together.
    """
    
    def __init__(
        self,
        embed_model: Any,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95
    ):
        self.embed_model = embed_model
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        
        self.splitter = SemanticSplitterNodeParser(
            embed_model=embed_model,
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold
        )
        
        logger.info(
            f"Initialized SemanticChunking: buffer={buffer_size}, "
            f"threshold={breakpoint_percentile_threshold}"
        )
    
    def chunk_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[TextNode]:
        """Chunk documents using semantic similarity."""
        nodes = self.splitter.get_nodes_from_documents(documents)
        logger.info(f"Created {len(nodes)} chunks using SemanticChunking")
        return nodes
    
    def get_name(self) -> str:
        return "semantic"


class SentenceChunkingStrategy(ChunkingStrategy):
    """
    Sentence-based chunking.
    Creates chunks at sentence boundaries for better coherence.
    """
    
    def __init__(
        self,
        sentences_per_chunk: int = 5,
        overlap_sentences: int = 1
    ):
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        
        # Use SentenceSplitter with sentence-based parameters
        self.splitter = SentenceSplitter(
            chunk_size=sentences_per_chunk * 100,  # Approximate
            chunk_overlap=overlap_sentences * 50,
            separator="."
        )
        
        logger.info(
            f"Initialized SentenceChunking: sentences={sentences_per_chunk}, "
            f"overlap={overlap_sentences}"
        )
    
    def chunk_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[TextNode]:
        """Chunk documents at sentence boundaries."""
        nodes = self.splitter.get_nodes_from_documents(documents)
        logger.info(f"Created {len(nodes)} chunks using SentenceChunking")
        return nodes
    
    def get_name(self) -> str:
        return "sentence"


class ChunkingStrategyFactory:
    """
    Factory for creating chunking strategies.
    Implements Factory pattern.
    """
    
    _strategies: Dict[str, type] = {
        "fixed_size": FixedSizeChunkingStrategy,
        "semantic": SemanticChunkingStrategy,
        "sentence": SentenceChunkingStrategy,
    }
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type) -> None:
        """Register a custom chunking strategy."""
        cls._strategies[name] = strategy_class
        logger.info(f"Registered chunking strategy: {name}")
    
    @classmethod
    def create_strategy(
        cls,
        strategy_name: str,
        **kwargs
    ) -> ChunkingStrategy:
        """
        Create chunking strategy instance.
        
        Args:
            strategy_name: Strategy identifier
            **kwargs: Strategy-specific parameters
        
        Returns:
            Chunking strategy instance
        
        Raises:
            ValueError: If strategy is not supported
        """
        if strategy_name not in cls._strategies:
            raise ValueError(
                f"Unsupported chunking strategy: {strategy_name}. "
                f"Available: {list(cls._strategies.keys())}"
            )
        
        strategy_class = cls._strategies[strategy_name]
        logger.info(f"Creating chunking strategy: {strategy_name}")
        return strategy_class(**kwargs)
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available strategy names."""
        return list(cls._strategies.keys())
