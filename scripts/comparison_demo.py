#!/usr/bin/env python3
"""
Demonstration script for the /compare endpoint.
Shows how different retrieval methods (contextual, BM25, TF-IDF) perform on the same query.
"""

import requests
import json
from typing import Dict, Any


def compare_retrieval_methods(query: str, k: int = 5, base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Compare different retrieval methods for a single query.
    
    Args:
        query: The question to ask
        k: Number of chunks to retrieve per method
        base_url: API base URL
        
    Returns:
        Comparison results from all methods
    """
    endpoint = f"{base_url}/api/v1/compare"
    
    payload = {
        "q": query,
        "k": k,
        "methods": ["contextual", "bm25", "tfidf"]
    }
    
    print(f"\n{'='*80}")
    print(f"Comparing Retrieval Methods")
    print(f"{'='*80}")
    print(f"Query: {query}")
    print(f"Top-K: {k}")
    print(f"{'='*80}\n")
    
    try:
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        
        # Display results for each method
        print(f"COMPARISON RESULTS\n")
        
        for result in data["results"]:
            print(f"\n{'─'*80}")
            print(f"Method: {result['method'].upper()}")
            print(f"{'─'*80}")
            print(f"Confidence: {result['confidence_score']:.3f} ({result['confidence_level']})")
            print(f"Latency: {result['latency_ms']:.2f}ms")
            print(f"Sources Retrieved: {result['num_sources']}")
            print(f"\nTop 3 Scores: {', '.join([f'{s:.3f}' for s in result['top_source_scores']])}")
            print(f"\nAnswer:\n{result['answer'][:300]}...")
            
            # Show top sources
            print(f"\nTop Sources:")
            for i, source in enumerate(result['sources'][:3], 1):
                print(f"  {i}. Page {source['page']} - Score: {source['score']:.3f}")
                print(f"     {source['content'][:150]}...")
        
        # Display summary
        print(f"\n\n{'='*80}")
        print(f"SUMMARY & RECOMMENDATIONS")
        print(f"{'='*80}")
        
        summary = data["summary"]
        print(f"Total Comparison Time: {data['total_latency_ms']:.2f}ms")
        print(f"Fastest Method: {summary['fastest_method']}")
        print(f"Highest Confidence: {summary['highest_confidence_method']}")
        print(f"Recommended Method: {summary['recommended_method']}")
        
        print(f"\nInsights:")
        for insight in summary.get('insights', []):
            print(f"  {insight}")
        
        print(f"\nConfidence Comparison:")
        for method, conf in summary['confidence_comparison'].items():
            print(f"  {method:12s}: {conf:.3f}")
        
        print(f"\nLatency Comparison:")
        for method, lat in summary['latency_comparison'].items():
            print(f"  {method:12s}: {lat:.2f}ms")
        
        print(f"\n{'='*80}\n")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f" Error calling API: {e}")
        return {}
    except Exception as e:
        print(f" Error: {e}")
        return {}


def main():
    """Run comparison demos with different query types."""
    
    # Sample queries from different domains
    queries = [
        "What is the main advantage of the Transformer architecture?",
        "How does BERT differ from GPT?",
        "What is contextual retrieval?",
    ]
    
    print("\n" + "="*80)
    print("RETRIEVAL METHOD COMPARISON DEMO")
    print("="*80)
    print("\nThis demo compares three retrieval methods:")
    print("  1. Contextual: Anthropic's contextual retrieval with semantic embeddings")
    print("  2. BM25: Traditional probabilistic keyword matching")
    print("  3. TF-IDF: Classical term frequency based retrieval")
    print("\nStarting API server at http://localhost:8000")
    print("="*80)
    
    for i, query in enumerate(queries, 1):
        print(f"\n\n{'#'*80}")
        print(f"# QUERY {i}/{len(queries)}")
        print(f"{'#'*80}")
        
        result = compare_retrieval_methods(query, k=5)
        
        if not result:
            print(f"  Skipping remaining queries due to error\n")
            break
        
        if i < len(queries):
            input("\nPress Enter to continue to next query...")
    
    print("\n Demo completed!\n")


if __name__ == "__main__":
    # Check if API is running
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        if response.status_code != 200:
            print("  Warning: API health check returned non-200 status")
    except requests.exceptions.RequestException:
        print("\n Error: Cannot connect to API at http://localhost:8000")
        print("Please start the API server first:")
        print("  python src/main.py\n")
        exit(1)
    
    main()
