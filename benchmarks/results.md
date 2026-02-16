# Contextual RAG Benchmark Report
 
**Ground Truth Queries:** 10  
**LLM Model:** mistral:latest  
**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2  
**Chunk Size:** 512  

## Executive Summary

This report presents comprehensive benchmarking results for different retrieval methods
implemented in the Contextual RAG system. Metrics include query latency, semantic
similarity to ground truth answers, and retrieval recall.

## Results by Method

### HYBRID

| Metric | Value |
|--------|-------|
| **Average Latency** | 51.283 seconds |
| **P95 Latency** | 67.697 seconds |
| **Avg Semantic Similarity** | 0.3890 |
| **Recall@1** | 0.0000 |
| **Recall@3** | 0.2000 |
| **Recall@5** | 0.3000 |
| **Total Queries** | 10 |

### TFIDF

| Metric | Value |
|--------|-------|
| **Average Latency** | 61.284 seconds |
| **P95 Latency** | 73.918 seconds |
| **Avg Semantic Similarity** | 0.1919 |
| **Recall@1** | 0.2000 |
| **Recall@3** | 0.4000 |
| **Recall@5** | 0.5000 |
| **Total Queries** | 10 |

### BM25

| Metric | Value |
|--------|-------|
| **Average Latency** | 62.391 seconds |
| **P95 Latency** | 72.891 seconds |
| **Avg Semantic Similarity** | 0.1281 |
| **Recall@1** | 0.1000 |
| **Recall@3** | 0.4000 |
| **Recall@5** | 0.4000 |
| **Total Queries** | 10 |

### CONTEXTUAL

| Metric | Value |
|--------|-------|
| **Average Latency** | 13.780 seconds |
| **P95 Latency** | 31.425 seconds |
| **Avg Semantic Similarity** | 0.0628 |
| **Recall@1** | 0.0000 |
| **Recall@3** | 0.0000 |
| **Recall@5** | 0.0000 |
| **Total Queries** | 10 |

## Method Comparison

| Method | Avg Latency (s) | Semantic Similarity | Recall@5 |
|--------|------------------|---------------------|----------|
| hybrid | 51.283 | 0.3890 | 0.3000 |
| tfidf | 61.284 | 0.1919 | 0.5000 |
| bm25 | 62.391 | 0.1281 | 0.4000 |
| contextual | 13.780 | 0.0628 | 0.0000 |

## Analysis

### Latency Performance

The latency metrics show the time taken to process each query, including both retrieval
and answer generation. Lower values indicate faster response times.

### Semantic Similarity

Semantic similarity measures how close the generated answers are to the ground truth
answers using cosine similarity of embeddings. Higher values (closer to 1.0) indicate
better answer quality.

### Recall Metrics

Recall@K measures the proportion of relevant documents retrieved in the top K results.
Higher recall indicates better retrieval effectiveness.

## Recommendations

Based on the benchmark results:

1. **Best Overall**: The method with highest semantic similarity provides the best answer quality
2. **Fastest**: The method with lowest latency is best for latency-sensitive applications
3. **Balanced**: Hybrid method typically offers good balance between quality and performance

## Configuration

```json
{
    "llm_model": "mistral:latest",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k": 5
}
```

## Methodology

All benchmarks were run on the same hardware with identical configurations. Each query
was executed sequentially to avoid resource contention. Metrics represent averages
across all ground truth queries.

