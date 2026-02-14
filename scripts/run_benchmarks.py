"""
Benchmarking and evaluation script.
Measures latency, semantic similarity, and recall metrics.
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

from config.settings import get_settings
from src.core.query_engine import QueryEngine
from src.models.schemas import GroundTruthQA, BenchmarkResult
from src.utils.logger import setup_logger
from src.utils.metrics import cosine_similarity, calculate_recall_at_k

logger = setup_logger(__name__)


class BenchmarkRunner:
    """
    Runs comprehensive benchmarks on the RAG system.
    Evaluates multiple retrieval methods against ground truth.
    """
    
    def __init__(
        self,
        query_engine: QueryEngine,
        ground_truth_path: str
    ):
        """
        Initialize benchmark runner.
        
        Args:
            query_engine: Initialized query engine
            ground_truth_path: Path to ground truth JSON file
        """
        self.query_engine = query_engine
        self.ground_truth = self._load_ground_truth(ground_truth_path)
        self.embed_model = query_engine.embed_model
        
        logger.info(f"Initialized BenchmarkRunner with {len(self.ground_truth)} QA pairs")
    
    def _load_ground_truth(self, path: str) -> List[GroundTruthQA]:
        """Load ground truth QA pairs from JSON."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            qa_pairs = [GroundTruthQA(**item) for item in data]
            logger.info(f"Loaded {len(qa_pairs)} ground truth QA pairs")
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            raise
    
    def run_benchmark(
        self,
        methods: List[str] = None,
        top_k_values: List[int] = None
    ) -> Dict[str, BenchmarkResult]:
        """
        Run comprehensive benchmark across methods.
        
        Args:
            methods: List of retrieval methods to benchmark
            top_k_values: List of K values for recall calculation
        
        Returns:
            Dictionary of method name to benchmark results
        """
        methods = methods or ["contextual", "bm25", "tfidf", "hybrid"]
        top_k_values = top_k_values or [1, 3, 5]
        
        logger.info(f"Starting benchmark for methods: {methods}")
        results = {}
        
        for method in methods:
            logger.info(f"Benchmarking method: {method}")
            result = self._benchmark_method(method, top_k_values)
            results[method] = result
        
        logger.info("Benchmark completed")
        return results
    
    def _benchmark_method(
        self,
        method: str,
        top_k_values: List[int]
    ) -> BenchmarkResult:
        """Benchmark a single retrieval method."""
        latencies = []
        similarities = []
        recalls = {k: [] for k in top_k_values}
        
        for qa in self.ground_truth:
            try:
                # Measure latency
                start_time = time.perf_counter()
                answer, sources, stats = self.query_engine.query(
                    query_text=qa.question,
                    top_k=max(top_k_values),
                    method=method
                )
                latency = (time.perf_counter() - start_time) * 1000
                latencies.append(latency)
                
                # Calculate semantic similarity
                generated_embedding = self.embed_model.get_text_embedding(answer)
                ground_truth_embedding = self.embed_model.get_text_embedding(qa.answer)
                similarity = cosine_similarity(
                    np.array(generated_embedding),
                    np.array(ground_truth_embedding)
                )
                similarities.append(similarity)
                
                # Calculate recall@K
                retrieved_pages = [s.page for s in sources if s.page is not None]
                relevant_pages = [qa.page]
                
                for k in top_k_values:
                    recall = calculate_recall_at_k(
                        [str(p) for p in retrieved_pages],
                        [str(p) for p in relevant_pages],
                        k
                    )
                    recalls[k].append(recall)
                
                logger.debug(
                    f"  Q: {qa.question[:50]}... | "
                    f"Latency: {latency:.2f}ms | "
                    f"Similarity: {similarity:.3f}"
                )
                
            except Exception as e:
                logger.error(f"Error benchmarking query '{qa.question}': {e}")
                continue
        
        # Aggregate results
        result = BenchmarkResult(
            method=method,
            avg_latency_ms=float(np.mean(latencies)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            avg_semantic_similarity=float(np.mean(similarities)),
            recall_at_1=float(np.mean(recalls[1])) if 1 in recalls else None,
            recall_at_3=float(np.mean(recalls[3])) if 3 in recalls else None,
            recall_at_5=float(np.mean(recalls[5])) if 5 in recalls else None,
            total_queries=len(latencies)
        )
        
        logger.info(
            f"  Results - Latency: {result.avg_latency_ms:.2f}ms (p95: {result.p95_latency_ms:.2f}ms) | "
            f"Similarity: {result.avg_semantic_similarity:.3f} | "
            f"Recall@5: {result.recall_at_5:.3f}"
        )
        
        return result
    
    def generate_report(
        self,
        results: Dict[str, BenchmarkResult],
        output_path: str = None
    ) -> str:
        """
        Generate markdown benchmark report.
        
        Args:
            results: Benchmark results
            output_path: Optional path to save report
        
        Returns:
            Markdown report string
        """
        settings = get_settings()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Contextual RAG Benchmark Report

**Generated:** {timestamp}  
**Ground Truth Queries:** {len(self.ground_truth)}  
**LLM Model:** {settings.llm_model}  
**Embedding Model:** {settings.embedding_model}  
**Chunk Size:** {settings.chunk_size}  

## Executive Summary

This report presents comprehensive benchmarking results for different retrieval methods
implemented in the Contextual RAG system. Metrics include query latency, semantic
similarity to ground truth answers, and retrieval recall.

## Results by Method

"""
        
        # Sort by avg semantic similarity (descending)
        sorted_methods = sorted(
            results.items(),
            key=lambda x: x[1].avg_semantic_similarity,
            reverse=True
        )
        
        for method, result in sorted_methods:
            report += f"""### {method.upper()}

| Metric | Value |
|--------|-------|
| **Average Latency** | {result.avg_latency_ms:.2f} ms |
| **P95 Latency** | {result.p95_latency_ms:.2f} ms |
| **Avg Semantic Similarity** | {result.avg_semantic_similarity:.4f} |
| **Recall@1** | {result.recall_at_1:.4f if result.recall_at_1 else 'N/A'} |
| **Recall@3** | {result.recall_at_3:.4f if result.recall_at_3 else 'N/A'} |
| **Recall@5** | {result.recall_at_5:.4f if result.recall_at_5 else 'N/A'} |
| **Total Queries** | {result.total_queries} |

"""
        
        # Comparison table
        report += """## Method Comparison

| Method | Avg Latency (ms) | Semantic Similarity | Recall@5 |
|--------|------------------|---------------------|----------|
"""
        
        for method, result in sorted_methods:
            report += f"| {method} | {result.avg_latency_ms:.2f} | {result.avg_semantic_similarity:.4f} | {result.recall_at_5:.4f if result.recall_at_5 else 'N/A'} |\n"
        
        report += """
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
    "llm_model": \"""" + settings.llm_model + """\",
    "embedding_model": \"""" + settings.embedding_model + """\",
    "chunk_size": """ + str(settings.chunk_size) + """,
    "chunk_overlap": """ + str(settings.chunk_overlap) + """,
    "top_k": """ + str(settings.top_k) + """
}
```

## Methodology

All benchmarks were run on the same hardware with identical configurations. Each query
was executed sequentially to avoid resource contention. Metrics represent averages
across all ground truth queries.

---

*Generated by Contextual RAG Benchmarking System*
"""
        
        # Save report if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"Report saved to: {output_path}")
            
            # Also save JSON results
            json_path = output_file.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {method: result.dict() for method, result in results.items()},
                    f,
                    indent=2
                )
            logger.info(f"JSON results saved to: {json_path}")
        
        return report


def main():
    """Main benchmark execution."""
    settings = get_settings()
    
    logger.info("=== Starting Benchmark ===")
    
    # Check if ground truth exists
    if not Path(settings.ground_truth_path).exists():
        logger.error(f"Ground truth file not found: {settings.ground_truth_path}")
        logger.error("Please run generate_ground_truth.py first")
        return
    
    # Check if PDF exists
    if not Path(settings.pdf_path).exists():
        logger.error(f"PDF file not found: {settings.pdf_path}")
        return
    
    # Initialize query engine
    logger.info("Initializing query engine...")
    query_engine = QueryEngine(
        pdf_path=settings.pdf_path,
        chunking_strategy="fixed_size",
        enable_contextual_retrieval=True
    )
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(
        query_engine=query_engine,
        ground_truth_path=settings.ground_truth_path
    )
    
    # Run benchmarks
    results = runner.run_benchmark(
        methods=["contextual", "bm25", "tfidf", "hybrid"],
        top_k_values=[1, 3, 5]
    )
    
    # Generate report
    report = runner.generate_report(
        results=results,
        output_path=settings.benchmark_results_path.replace('.json', '.md')
    )
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    for method, result in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Avg Latency: {result.avg_latency_ms:.2f}ms")
        print(f"  Semantic Similarity: {result.avg_semantic_similarity:.4f}")
        print(f"  Recall@5: {result.recall_at_5:.4f if result.recall_at_5 else 'N/A'}")
    print("\n" + "="*80)
    
    logger.info("=== Benchmark Complete ===")


if __name__ == "__main__":
    main()
