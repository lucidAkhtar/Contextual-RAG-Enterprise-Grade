# Architecture Diagram Generation Script
# Run this to create a visual architecture diagram

from diagrams import Diagram, Cluster, Edge
from diagrams.programming.framework import Fastapi
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.inmemory import Redis
from diagrams.programming.language import Python
from diagrams.custom import Custom

# Note: requires 'diagrams' package and graphviz
# uv add  diagrams
# brew install graphviz (on macOS)

with Diagram("Contextual RAG Architecture", show=False, direction="TB"):
    with Cluster("API Layer"):
        api = Fastapi("REST API\n(OpenAPI)")
    
    with Cluster("Core Engine"):
        engine = Python("Query Engine\n(Orchestrator)")
    
    with Cluster("Retrieval Layer"):
        with Cluster("Retrievers"):
            contextual = Python("Contextual\nRetriever")
            bm25 = Python("BM25\nRetriever")
            tfidf = Python("TF-IDF\nRetriever")
        
        hybrid = Python("Hybrid Fusion\n(RRF)")
    
    with Cluster("Storage & Models"):
        vector_db = PostgreSQL("ChromaDB\nVector Store")
        embeddings = Python("HuggingFace\nEmbeddings")
        llm = Python("Ollama\nLLM")
    
    # Connections
    api >> engine
    engine >> [contextual, bm25, tfidf]
    [contextual, bm25, tfidf] >> hybrid
    contextual >> vector_db
    contextual >> embeddings
    hybrid >> llm
