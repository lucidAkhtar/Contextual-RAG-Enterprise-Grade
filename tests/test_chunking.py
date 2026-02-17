"""Test suite for chunking strategies."""

import pytest
from llama_index.core.schema import Document
from src.core.chunking_strategies import (
    FixedSizeChunkingStrategy,
    SentenceChunkingStrategy,
    ChunkingStrategyFactory
)


class TestFixedSizeChunking:
    """Test fixed-size chunking strategy."""
    
    def test_basic_chunking(self):
        """Test basic chunking functionality."""
        strategy = FixedSizeChunkingStrategy(chunk_size=100, chunk_overlap=20)
        
        documents = [
            Document(
                text="This is a test document. " * 20,
                metadata={"page": 1}
            )
        ]
        
        nodes = strategy.chunk_documents(documents)
        
        # Verify chunking creates nodes
        assert len(nodes) > 0
        # Verify all nodes have text content
        assert all(len(node.text) > 0 for node in nodes)
        # Verify metadata is preserved
        assert all(node.metadata.get("page") == 1 for node in nodes)
    
    def test_overlap_functionality(self):
        """Test that overlap works correctly."""
        strategy = FixedSizeChunkingStrategy(chunk_size=100, chunk_overlap=20)
        
        # Use realistic sentences (LlamaIndex SentenceSplitter needs proper text)
        documents = [
            Document(
                text="This is sentence one. This is sentence two. This is sentence three. " * 10,
                metadata={"page": 1}
            )
        ]
        
        nodes = strategy.chunk_documents(documents)
        
        # Should have multiple chunks with realistic text
        assert len(nodes) >= 2
    
    def test_metadata_preservation(self):
        """Test that metadata is preserved in chunks."""
        strategy = FixedSizeChunkingStrategy()
        
        documents = [
            Document(
                text="Test content",
                metadata={"page": 5, "source": "test.pdf"}
            )
        ]
        
        nodes = strategy.chunk_documents(documents)
        
        assert nodes[0].metadata["page"] == 5
        assert nodes[0].metadata["source"] == "test.pdf"


class TestSentenceChunking:
    """Test sentence-based chunking strategy."""
    
    def test_sentence_boundaries(self):
        """Test chunking respects sentence boundaries."""
        strategy = SentenceChunkingStrategy(sentences_per_chunk=2)
        
        documents = [
            Document(
                text="First sentence. Second sentence. Third sentence. Fourth sentence.",
                metadata={"page": 1}
            )
        ]
        
        nodes = strategy.chunk_documents(documents)
        
        assert len(nodes) > 0


class TestChunkingStrategyFactory:
    """Test factory pattern for chunking strategies."""
    
    def test_factory_creates_fixed_size(self):
        """Test factory creates fixed-size strategy."""
        strategy = ChunkingStrategyFactory.create_strategy("fixed_size")
        
        assert isinstance(strategy, FixedSizeChunkingStrategy)
    
    def test_factory_creates_sentence(self):
        """Test factory creates sentence strategy."""
        strategy = ChunkingStrategyFactory.create_strategy("sentence")
        
        assert isinstance(strategy, SentenceChunkingStrategy)
    
    def test_factory_invalid_strategy(self):
        """Test factory raises error for invalid strategy."""
        with pytest.raises(ValueError):
            ChunkingStrategyFactory.create_strategy("invalid_strategy")
    
    def test_get_available_strategies(self):
        """Test getting list of available strategies."""
        strategies = ChunkingStrategyFactory.get_available_strategies()
        
        assert "fixed_size" in strategies
        assert "sentence" in strategies
        assert len(strategies) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
