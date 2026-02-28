# LlamaIndex Internals: Query Engine & Retrievers Deep Dive

##  Table of Contents
1. [What is LlamaIndex?](#what-is-llamaindex)
2. [Core Concept: The TextNode](#core-concept-the-textnode)
3. [Document Ingestion Pipeline](#document-ingestion-pipeline)
4. [Query Execution Flow](#query-execution-flow)
5. [QueryEngine.py Internals](#queryenginepy-internals)
6. [Retrievers.py Internals](#retrieverspy-internals)
7. [Complete Request-Response Cycle](#complete-request-response-cycle)

---

##  What is LlamaIndex?

**LlamaIndex** (formerly GPT Index) is a data framework for connecting custom data sources to Large Language Models (LLMs).

### Core Philosophy
```
Raw Data → Structure → Index → Query → LLM → Answer
```

### Key Abstractions Used in This Project

| Component | Purpose | Our Usage |
|-----------|---------|-----------|
| **Document** | Raw text with metadata | Page-level PDF content |
| **TextNode** | Chunk of text (core unit) | 512-token chunks with metadata |
| **VectorStoreIndex** | Embedding storage & retrieval | ChromaDB integration |
| **BaseRetriever** | Search interface | Custom contextual retriever |
| **QueryBundle** | Query wrapper | Holds query string + metadata |
| **NodeWithScore** | Retrieved node + relevance | Results from vector search |
| **BaseEmbedding** | Embedding model interface | HuggingFace embeddings |
| **LLM** | Language model interface | Ollama/Mistral integration |

---

##  Core Concept: The TextNode

### What is a TextNode?

A **TextNode** is LlamaIndex's fundamental unit of information. Think of it as a **smart container** for a chunk of text with rich metadata.

```python
from llama_index.core.schema import TextNode

# Simplified representation:
class TextNode:
    node_id: str           # Unique identifier (UUID)
    text: str              # The actual text content
    metadata: Dict         # Key-value pairs of metadata
    embedding: List[float] # Vector representation (optional)
    
    def get_content() -> str:
        """Returns the text content"""
        return self.text
```

### Real Example from Our System

```python
# After chunking a PDF page
node = TextNode(
    node_id="a7f3c21d-8e9b-4a2c-9d1e-5f8a6b3c2d1e",
    text="The Transformer architecture can parallelize computation by relying entirely on attention mechanisms...",
    metadata={
        "page": 8,
        "source": "data/attention_is_all_you_need.pdf",
        "source_document": "attention_is_all_you_need.pdf",
        "total_pages": 15,
        "chunk_id": 45,
        # After contextual enrichment:
        "contextual_prefix": "Transformer model parallelization advantages: attention mechanisms eliminate recurrence...",
        "original_text": "The Transformer architecture can parallelize..."
    },
    embedding=[0.23, -0.45, 0.12, ..., 0.67]  # 384 dimensions
)

# Access methods:
node.get_content()  # Returns text
node.node_id        # UUID string
node.metadata["page"]  # 8
```

### Why TextNode Instead of Plain Text?

| Problem | Plain Text | TextNode |
|---------|------------|----------|
| Track source |  Lost |  In metadata |
| Find similar |  Re-compute |  Cached embedding |
| Reference back |  Impossible |  node_id tracking |
| Add context |  Modify string |  Metadata field |
| Type safety |  String only |  Structured object |

---

##  Document Ingestion Pipeline

### Step-by-Step: PDF → TextNodes → VectorStore

Let me trace through **exact code flow** when the system starts:

#### **Step 1: Load PDF with DocumentProcessor**

```python
# query_engine.py line 208-210
documents = self.doc_processor.load_pdf(pdf_path)

# What happens in document_processor.py:
def load_pdf(self, pdf_path: str) -> List[Document]:
    doc = fitz.open(pdf_path)  # PyMuPDF
    documents = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        cleaned_text = self._clean_text(text)
        
        # Create LlamaIndex Document
        documents.append(
            Document(
                text=cleaned_text,
                metadata={
                    "page": page_num + 1,
                    "source": str(path),
                    "total_pages": len(doc)
                }
            )
        )
    
    return documents  # One Document per page
```

**Result**: List of **Document** objects (one per PDF page)

```
[
  Document(text="Attention Is All You Need\n\nAbstract...", metadata={page: 1, ...}),
  Document(text="The Transformer model architecture...", metadata={page: 2, ...}),
  ...
]
```

---

#### **Step 2: Chunk Documents into TextNodes**

```python
# query_engine.py line 219-224
chunking_strategy = ChunkingStrategyFactory.create_strategy("fixed_size")
self.nodes = chunking_strategy.chunk_documents(all_documents)

# What happens in chunking_strategies.py:
class FixedSizeChunkingStrategy:
    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        # Uses LlamaIndex's SentenceSplitter
        self.splitter = SentenceSplitter(
            chunk_size=512,      # 512 tokens per chunk
            chunk_overlap=50     # 50 token overlap between chunks
        )
        
        nodes = self.splitter.get_nodes_from_documents(documents)
        # Returns List[TextNode]
        return nodes
```

**What `SentenceSplitter.get_nodes_from_documents()` does**:
1. **Tokenizes** each Document's text
2. **Splits** into 512-token chunks with 50-token overlap
3. **Creates TextNode objects** with:
   - Text content
   - Unique `node_id` (auto-generated UUID)
   - Inherited metadata from parent Document
   - Relationships to neighboring chunks

**Result**: 132 **TextNode** objects

```
[
  TextNode(id="node_0", text="Attention Is All You Need Abstract...", metadata={page: 1}),
  TextNode(id="node_1", text="...state-of-the-art results...", metadata={page: 1}),
  TextNode(id="node_2", text="1 Introduction Recurrent neural...", metadata={page: 2}),
  ...
  TextNode(id="node_131", text="...future work and conclusions.", metadata={page: 15})
]
```

---

#### **Step 3: Enrich Nodes with Contextual Information**

```python
# query_engine.py line 243-244
if self.enable_contextual_retrieval:
    self._enrich_nodes_before_embedding()

# What happens:
def _enrich_nodes_before_embedding(self):
    for node in self.nodes:
        # Generate context using LLM
        contextual_prefix = llm.complete(prompt_about_chunk).text
        
        # CRITICAL: Modify the node's TEXT field
        node.metadata["contextual_prefix"] = contextual_prefix
        node.metadata["original_text"] = node.text  # Save original
        
        # Prepend context to text that will be embedded
        node.text = f"{contextual_prefix}\n\n{node.metadata['original_text']}"
```

**Before Enrichment**:
```python
node.text = "This approach achieved 28.4 BLEU score."
node.metadata = {page: 8}
```

**After Enrichment**:
```python
node.text = """Transformer model parallelization advantages: attention mechanisms eliminate recurrence, achieves 28.4 BLEU on WMT 2014 English-German translation.

This approach achieved 28.4 BLEU score."""

node.metadata = {
    page: 8,
    contextual_prefix: "Transformer model parallelization advantages...",
    original_text: "This approach achieved 28.4 BLEU score."
}
```

**Key Insight**: The `node.text` field now contains **enriched text**, which will be embedded!

---

#### **Step 4: Create Vector Store Index (Embedding)**

```python
# query_engine.py line 247
self._init_vector_store()

# What happens:
def _init_vector_store(self):
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("contextual_rag")
    
    # Wrap ChromaDB in LlamaIndex abstraction
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create VectorStoreIndex from nodes
    self.vector_store_index = VectorStoreIndex(
        nodes=self.nodes,              # Our 132 TextNodes
        storage_context=storage_context,
        embed_model=self.embed_model,  # HuggingFace embeddings
        show_progress=True
    )
```

**What `VectorStoreIndex()` constructor does**:
1. **For each TextNode**:
   - Extracts `node.text` (enriched text!)
   - Calls `embed_model.get_text_embedding(node.text)` → 384-dim vector
   - Stores in ChromaDB: `{node_id: embedding_vector}`
   - Stores in docstore: `{node_id: TextNode object}`

2. **Result**: Two storage systems
   - **Vector Store (ChromaDB)**: Maps node_id → embedding vector
   - **Document Store (in-memory)**: Maps node_id → full TextNode object

---

#### **Step 5: Initialize Retrievers**

```python
# query_engine.py line 250
self._init_retrievers()

# What happens:
def _init_retrievers(self):
    # Pass nodes and index to retrievers
    self.contextual_retriever = ContextualRetriever(
        nodes=self.nodes,                    # All 132 TextNode objects
        embed_model=self.embed_model,        # For query embedding
        vector_store_index=self.vector_store_index,  # For similarity search
        similarity_top_k=5
    )
    
    self.bm25_retriever = BM25Retriever(nodes=self.nodes)
    self.tfidf_retriever = TFIDFRetriever(nodes=self.nodes)
    
    self.hybrid_retriever = HybridRetriever(
        contextual_retriever=self.contextual_retriever,
        bm25_retriever=self.bm25_retriever,
        tfidf_retriever=self.tfidf_retriever
    )
```

**System is now ready to accept queries!**

---

##  Query Execution Flow

### The Complete Journey of a Query

When you send: `"What is the main advantage of the Transformer architecture?"`

---

### **Phase 1: API Request → Query Engine**

```python
# routes.py (FastAPI)
@router.post("/query")
async def query_endpoint(request: QueryRequest):
    answer, sources, stats = query_engine.query(
        query_text=request.q,
        top_k=request.k,
        method=request.retrieval_method  # "contextual"
    )
```

---

### **Phase 2: Query Method Dispatches to Retriever**

```python
# query_engine.py line 489-492
def query(self, query_text: str, top_k: int = 5, method: str = "contextual"):
    # Retrieve relevant chunks
    sources = self._retrieve(query_text, top_k, method)
    # Generate answer
    answer = self._generate_answer(query_text, sources)
    return answer, sources, stats

# query_engine.py line 625-640
def _retrieve(self, query: str, top_k: int, method: str) -> List[RetrievalSource]:
    if method == "contextual":
        query_bundle = QueryBundle(query_str=query)
        results = self.contextual_retriever._retrieve(query_bundle)
        
        # Convert NodeWithScore to RetrievalSource
        for node_with_score in results:
            sources.append(RetrievalSource(
                chunk_id=node_with_score.node.node_id,
                content=node_with_score.node.metadata.get("original_text"),
                score=node_with_score.score,
                page=node_with_score.node.metadata.get("page"),
                ...
            ))
```

**Key Objects**:
- **QueryBundle**: LlamaIndex wrapper around query string
- **NodeWithScore**: Contains retrieved TextNode + similarity score
- **RetrievalSource**: Our custom schema for API responses

---

### **Phase 3: ContextualRetriever Searches Vector Store**

```python
# retrievers.py line 232-266
class ContextualRetriever(BaseRetriever):
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Step 1: Embed the query
        query_embedding = self.embed_model.get_query_embedding(
            query_bundle.query_str
        )
        # Result: [0.23, -0.15, 0.42, ...] (384 dims)
        
        # Step 2: Query vector store
        query_obj = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=5
        )
        
        query_result = self.vector_store_index.vector_store.query(query_obj)
        # Result: {ids: [...], similarities: [...]}
        
        # Step 3: Fetch full TextNode objects from docstore
        nodes_with_scores = []
        for node_id, similarity in zip(query_result.ids, query_result.similarities):
            node = self.vector_store_index.docstore.get_document(node_id)
            nodes_with_scores.append(
                NodeWithScore(node=node, score=similarity)
            )
        
        return nodes_with_scores
```

#### What Happens in Each Step:

**Step 1: Query Embedding**
```python
query_text = "What is the main advantage of the Transformer architecture?"
embedding = embed_model.get_query_embedding(query_text)
# Uses sentence-transformers/all-MiniLM-L6-v2
# Returns: [0.234, -0.152, 0.421, ..., 0.089]  (384 floats)
```

**Step 2: Vector Similarity Search**
```python
# ChromaDB performs cosine similarity search:
# For each stored embedding:
#   similarity = cosine(query_embedding, stored_embedding)
# 
# Returns top-5 node_ids by similarity:
{
    "ids": ["node_45", "node_46", "node_7", "node_102", "node_23"],
    "similarities": [0.89, 0.87, 0.84, 0.81, 0.78]
}
```

**Step 3: Docstore Lookup**
```python
# For each node_id, fetch the full TextNode object
node_45 = docstore.get_document("node_45")
# Returns:
TextNode(
    node_id="node_45",
    text="Transformer model parallelization...\n\nThis approach achieved 28.4 BLEU",
    metadata={
        "page": 8,
        "source_document": "attention_is_all_you_need.pdf",
        "contextual_prefix": "Transformer model...",
        "original_text": "This approach achieved 28.4 BLEU"
    },
    embedding=[...]
)

# Wrap in NodeWithScore
NodeWithScore(node=node_45, score=0.89)
```

**Final Result**: List of 5 `NodeWithScore` objects

---

### **Phase 4: Generate Answer with LLM**

```python
# query_engine.py line 702-730
def _generate_answer(self, query: str, sources: List[RetrievalSource]) -> str:
    # Build context from retrieved nodes
    context_parts = []
    for idx, source in enumerate(sources[:5], 1):
        context_parts.append(
            f"[Source {idx} - {source.source_document}, page {source.page}]\n"
            f"{source.content}\n"
        )
    
    context = "\n\n".join(context_parts)
    
    # Create prompt
    prompt = f"""You are a helpful AI assistant.

Context:
{context}

Question: {query}

Answer based ONLY on the context above.

Answer:"""
    
    # Generate with LLM
    response = self.llm.complete(prompt)
    return response.text.strip()
```

**What `self.llm.complete()` does**:
1. Sends prompt to Ollama API (Mistral 7B)
2. LLM generates text token-by-token
3. Returns complete answer

---

##  QueryEngine.py Internals

### Class Structure & Responsibilities

```python
class QueryEngine:
    """Main orchestrator of the RAG pipeline"""
    
    # === Injected Dependencies ===
    llm: LLM                      # Language model (Ollama/Mistral)
    embed_model: BaseEmbedding    # Embedding model (HuggingFace)
    doc_processor: DocumentProcessor
    
    # === Core Data Structures ===
    nodes: List[TextNode]         # All 132 chunks (in memory)
    vector_store_index: VectorStoreIndex  # ChromaDB wrapper
    
    # === Retrievers ===
    contextual_retriever: ContextualRetriever
    bm25_retriever: BM25Retriever
    tfidf_retriever: TFIDFRetriever
    hybrid_retriever: HybridRetriever
```

### Key Methods Breakdown

#### **1. Ingestion Phase**

```python
def ingest_documents(self, pdf_paths: List[str]):
    """
    Entry point for document loading
    
    Flow:
    1. Load PDFs → List[Document]
    2. Chunk Documents → List[TextNode]
    3. Enrich nodes → Modify TextNode.text
    4. Embed nodes → Store in vector store
    5. Initialize retrievers
    """
```

#### **2. Query Phase**

```python
def query(self, query_text: str, top_k: int, method: str):
    """
    Entry point for queries
    
    Flow:
    1. Retrieve chunks via _retrieve()
    2. Generate answer via _generate_answer()
    3. Calculate confidence
    4. Return (answer, sources, stats)
    """
```

#### **3. Private Helper: _retrieve()**

```python
def _retrieve(self, query: str, top_k: int, method: str) -> List[RetrievalSource]:
    """
    Dispatch to appropriate retriever based on method
    
    Converts: List[NodeWithScore] → List[RetrievalSource]
    
    RetrievalSource contains:
    - chunk_id (node_id)
    - content (original_text from metadata)
    - score (similarity score)
    - page, source_document (from metadata)
    """
```

#### **4. Private Helper: _generate_answer()**

```python
def _generate_answer(self, query: str, sources: List[RetrievalSource]) -> str:
    """
    Build prompt from sources and query LLM
    
    Prompt structure:
    [Context from 5 sources]
    [Question]
    [Instructions]
    """
```

---

##  Retrievers.py Internals

### ContextualRetriever Deep Dive

```python
class ContextualRetriever(BaseRetriever):
    """
    Inherits from LlamaIndex's BaseRetriever
    Implements _retrieve() method required by interface
    """
    
    def __init__(
        self,
        nodes: List[TextNode],              # All chunks (for fallback)
        embed_model: BaseEmbedding,         # For query embedding
        vector_store_index: VectorStoreIndex,  # For search
        similarity_top_k: int = 5
    ):
        # Store references to system components
        self.nodes = nodes
        self.embed_model = embed_model
        self.vector_store_index = vector_store_index
        self.similarity_top_k = similarity_top_k
```

### How Vector Search Works

```python
def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
    # === Step 1: Embed Query ===
    query_embedding = self.embed_model.get_query_embedding(
        query_bundle.query_str
    )
    # HuggingFace model converts text → 384-dim vector
    
    # === Step 2: Create VectorStoreQuery ===
    query_obj = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=self.similarity_top_k
    )
    # This is LlamaIndex's abstraction for vector queries
    
    # === Step 3: Search Vector Store ===
    query_result = self.vector_store_index.vector_store.query(query_obj)
    # ChromaDB performs:
    #   - HNSW approximate nearest neighbor search
    #   - Cosine similarity metric
    #   - Returns top-k node IDs and scores
    
    # === Step 4: Fetch Full Nodes ===
    nodes_with_scores = []
    for node_id, similarity in zip(query_result.ids, query_result.similarities):
        node = self.vector_store_index.docstore.get_document(node_id)
        # Fetches the complete TextNode object
        
        nodes_with_scores.append(
            NodeWithScore(node=node, score=similarity)
        )
    
    return nodes_with_scores
```

### LlamaIndex Abstractions Used

#### **VectorStoreIndex**
```python
# Manages two stores:
vector_store_index.vector_store  # ChromaDB - stores embeddings
vector_store_index.docstore      # In-memory - stores TextNode objects

# Methods:
.vector_store.query(query_obj)   # Vector similarity search
.docstore.get_document(node_id)  # Fetch TextNode by ID
.docstore.add_documents([node])  # Add/update TextNode
```

#### **VectorStoreQuery**
```python
# Query configuration object
VectorStoreQuery(
    query_embedding=[...],    # Query vector
    similarity_top_k=5,       # Number of results
    filters=None              # Optional metadata filters
)
```

#### **NodeWithScore**
```python
# Search result wrapper
NodeWithScore(
    node=TextNode(...),       # The retrieved chunk
    score=0.89                # Similarity score (0-1)
)
```

---

##  Complete Request-Response Cycle

### Putting It All Together: From Request to Response

Let me trace **every single object transformation** when you query:

**Input**: HTTP Request
```json
POST /api/v1/query
{
  "q": "What is the main advantage of the Transformer architecture?",
  "k": 5,
  "retrieval_method": "contextual"
}
```

---

### **Step 1: API Layer → Query Engine**

```python
# routes.py
@router.post("/query")
async def query_endpoint(request: QueryRequest):
    # Pydantic validates and parses request
    query_text = request.q
    top_k = request.k
    method = request.retrieval_method
    
    # Call query engine
    answer, sources, stats = query_engine.query(
        query_text=query_text,
        top_k=top_k,
        method=method
    )
```

**Object**: `QueryRequest` (Pydantic model) → method parameters

---

### **Step 2: Query Engine → Retriever**

```python
# query_engine.py - query() method
def query(self, query_text: str, top_k: int, method: str):
    sources = self._retrieve(query_text, top_k, method)
    
# _retrieve() dispatches to retriever
def _retrieve(self, query: str, top_k: int, method: str):
    if method == "contextual":
        # Create QueryBundle (LlamaIndex wrapper)
        query_bundle = QueryBundle(query_str=query)
        
        # Call retriever
        results = self.contextual_retriever._retrieve(query_bundle)
        # results: List[NodeWithScore]
```

**Object**: `str` → `QueryBundle` → `List[NodeWithScore]`

---

### **Step 3: Retriever → Vector Store**

```python
# retrievers.py - ContextualRetriever._retrieve()
def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
    # 3.1: Embed query
    query_embedding = self.embed_model.get_query_embedding(
        query_bundle.query_str
    )
    # Type: List[float] (384 dimensions)
    
    # 3.2: Create vector query
    query_obj = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=5
    )
    # Type: VectorStoreQuery
    
    # 3.3: Search ChromaDB
    query_result = self.vector_store_index.vector_store.query(query_obj)
    # Type: VectorStoreQueryResult
    # Contains: ids=["node_45", "node_46", ...], 
    #           similarities=[0.89, 0.87, ...]
```

**Object**: `QueryBundle` → `List[float]` → `VectorStoreQuery` → `VectorStoreQueryResult`

---

### **Step 4: Docstore Lookup**

```python
    # 3.4: Fetch full TextNode objects
    nodes_with_scores = []
    for node_id, similarity in zip(query_result.ids, query_result.similarities):
        # Fetch from document store
        node = self.vector_store_index.docstore.get_document(node_id)
        # Type: TextNode
        
        # Wrap with score
        nodes_with_scores.append(
            NodeWithScore(node=node, score=similarity)
        )
    
    return nodes_with_scores
```

**Object**: `node_id` → `TextNode` → `NodeWithScore`

---

### **Step 5: Convert to API Schema**

```python
# query_engine.py - _retrieve() continues
for node_with_score in results:
    sources.append(RetrievalSource(
        chunk_id=node_with_score.node.node_id,
        content=node_with_score.node.metadata.get("original_text"),
        score=node_with_score.score,
        page=node_with_score.node.metadata.get("page"),
        source_document=node_with_score.node.metadata.get("source_document"),
        method="contextual",
        metadata=node_with_score.node.metadata
    ))

return sources  # List[RetrievalSource]
```

**Object**: `List[NodeWithScore]` → `List[RetrievalSource]`

---

### **Step 6: Generate Answer**

```python
# query_engine.py - query() continues
answer = self._generate_answer(query_text, sources)

def _generate_answer(self, query: str, sources: List[RetrievalSource]) -> str:
    # Build context string from sources
    context = "\n\n".join([
        f"[Source {i}]\n{source.content}"
        for i, source in enumerate(sources, 1)
    ])
    
    # Build prompt
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    
    # Call LLM
    response = self.llm.complete(prompt)
    # Type: CompletionResponse
    
    return response.text.strip()  # Type: str
```

**Object**: `List[RetrievalSource]` → `str (prompt)` → `CompletionResponse` → `str (answer)`

---

### **Step 7: Return to API**

```python
# query_engine.py - query() completes
return answer, sources, stats

# routes.py - query_endpoint() continues
return QueryResponse(
    answer=answer,
    sources=[...],
    confidence=confidence_score,
    retrieval_time_ms=82,
    generation_time_ms=4521,
    total_time_ms=4603
)
```

**Output**: HTTP Response
```json
{
  "answer": "The main advantage of the Transformer architecture over recurrent models is its ability to parallelize computation...",
  "sources": [
    {
      "chunk_id": "node_45",
      "content": "This approach achieved 28.4 BLEU score...",
      "score": 0.89,
      "page": 8,
      "source_document": "attention_is_all_you_need.pdf"
    },
    // 4 more sources...
  ],
  "confidence": 0.89,
  "confidence_level": "high",
  "retrieval_time_ms": 82,
  "generation_time_ms": 4521,
  "total_time_ms": 4603
}
```

---

##  Object Transformation Summary

```
HTTP Request (JSON)
    ↓
QueryRequest (Pydantic)
    ↓
str (query_text)
    ↓
QueryBundle (LlamaIndex)
    ↓
List[float] (query embedding)
    ↓
VectorStoreQuery (LlamaIndex)
    ↓
VectorStoreQueryResult (node_ids + scores)
    ↓
List[TextNode] (from docstore)
    ↓
List[NodeWithScore] (nodes + scores)
    ↓
List[RetrievalSource] (our schema)
    ↓
str (context for LLM)
    ↓
CompletionResponse (LLM output)
    ↓
QueryResponse (Pydantic)
    ↓
HTTP Response (JSON)
```

---

##  Key Takeaways

### 1. **TextNode is the Core Unit**
- Stores text, metadata, and embeddings
- Flows through entire pipeline
- Allows rich tracking of source information

### 2. **LlamaIndex Separates Concerns**
- **Vector Store**: Embeddings for fast search
- **Document Store**: Full TextNode objects for retrieval
- **Retrievers**: Search interface abstraction
- **Indexes**: Coordinate stores and embeddings

### 3. **Why We Use LlamaIndex**
-  **Abstraction**: Don't deal with raw ChromaDB API
-  **Standardization**: Common interfaces (BaseRetriever, BaseEmbedding)
-  **Flexibility**: Easy to swap vector stores (ChromaDB → Pinecone)
-  **Rich Metadata**: TextNode carries context through pipeline
-  **Node Relationships**: Automatic tracking of chunk relationships

### 4. **Critical Design Decisions**
```python
#  Good: Enrich before embedding
node.text = f"{contextual_prefix}\n\n{original_text}"
embedding = embed_model.get_text_embedding(node.text)

#  Bad: Enrich after embedding
embedding = embed_model.get_text_embedding(original_text)
# (too late - embedding doesn't have context!)
```

### 5. **Two Storage Systems**
- **Vector Store (ChromaDB)**: Fast similarity search
- **Document Store (in-memory dict)**: Full node retrieval
- **Critical**: Both must be synchronized (node_id as key)

---

##  Practical Examples

### Example 1: Debug What TextNode Contains

```python
# In retrievers.py, add logging:
def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
    query_result = self.vector_store_index.vector_store.query(query_obj)
    
    for node_id, similarity in zip(query_result.ids, query_result.similarities):
        node = self.vector_store_index.docstore.get_document(node_id)
        
        # DEBUG: Print what's in the node
        print(f"Node ID: {node.node_id}")
        print(f"Text length: {len(node.text)}")
        print(f"Metadata keys: {node.metadata.keys()}")
        print(f"Page: {node.metadata.get('page')}")
        print(f"Has contextual_prefix: {'contextual_prefix' in node.metadata}")
```

### Example 2: Understand Query Embedding

```python
# In query_engine.py, add:
query_embedding = self.embed_model.get_query_embedding(query_text)
print(f"Query: {query_text}")
print(f"Embedding shape: {len(query_embedding)}")  # 384
print(f"First 5 dims: {query_embedding[:5]}")  # [0.23, -0.15, ...]
print(f"Embedding norm: {np.linalg.norm(query_embedding)}")  # ~1.0 (normalized)
```

### Example 3: See Vector Store vs Docstore

```python
# After ingestion:
print(f"Nodes in memory: {len(query_engine.nodes)}")  # 132
print(f"Vectors in ChromaDB: {query_engine.vector_store_index.vector_store._collection.count()}")  # 132
print(f"Docs in docstore: {len(query_engine.vector_store_index.docstore.docs)}")  # 132

# They should all match!
```

---

##  Summary for Interview/Demo

### Quick Talking Points

**"What is a TextNode?"**
> A TextNode is LlamaIndex's core data structure representing a chunk of text with metadata and optional embeddings. It's like a smart container that carries source information, page numbers, and contextual data through the entire RAG pipeline.

**"How does query_engine.py work?"**
> QueryEngine orchestrates the RAG pipeline. It loads PDFs, chunks them into TextNodes, enriches them with LLM-generated context, embeds them into ChromaDB, and provides a unified query interface that retrieves relevant chunks and generates answers using an LLM.

**"How do retrievers.py work?"**
> Retrievers implement the search logic. ContextualRetriever embeds the query, performs vector similarity search in ChromaDB to get node IDs, fetches full TextNode objects from the docstore, and returns them wrapped with similarity scores.

**"Why use LlamaIndex?"**
> LlamaIndex provides abstractions that separate concerns: TextNode for data, VectorStoreIndex for search, BaseRetriever for interfaces. This makes it easy to swap components (e.g., ChromaDB → Pinecone) without rewriting core logic. It standardizes the RAG pipeline.

---

**You're now equipped to explain every detail of the system!** 
