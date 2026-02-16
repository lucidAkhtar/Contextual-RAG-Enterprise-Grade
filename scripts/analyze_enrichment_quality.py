#!/usr/bin/env python3
"""Analyze contextual enrichment quality to diagnose poor retrieval performance."""

import json
from pathlib import Path

# Load enrichment cache
cache_path = Path("chroma_db/contextual_enrichment_cache.json")
with open(cache_path, 'r') as f:
    cache = json.load(f)

print('='*80)
print('CONTEXTUAL ENRICHMENT QUALITY ANALYSIS')
print('='*80)
print(f'\nTotal enriched chunks: {len(cache)}\n')

# Show detailed examples
print("="*80)
print("DETAILED EXAMPLES (First 5 chunks)")
print("="*80)

for i, (hash_key, entry) in enumerate(list(cache.items())[:5], 1):
    prefix = entry['contextual_prefix']
    original = entry['original_text']
    
    print(f"\n{'='*80}")
    print(f"Example {i} (hash: {hash_key})")
    print(f"{'='*80}")
    print(f"\nCONTEXTUAL PREFIX ({len(prefix)} chars):")
    print(f"{prefix}\n")
    print(f"ORIGINAL TEXT ({len(original)} chars):")
    print(f"{original[:300]}...")
    print(f"\nCOMBINED (what gets embedded):")
    combined = f"{prefix}\n\n{original[:200]}"
    print(f"{combined}...")

# Analyze prefix patterns
print("\n\n" + "="*80)
print("PREFIX PATTERN ANALYSIS")
print("="*80)

prefixes = [v['contextual_prefix'] for v in cache.values()]

# Count common starting patterns (first 3 words)
starts_with = {}
for prefix in prefixes:
    first_words = ' '.join(prefix.split()[:3])
    starts_with[first_words] = starts_with.get(first_words, 0) + 1

print(f'\nTop 10 Most Common Starting Patterns:')
sorted_patterns = sorted(starts_with.items(), key=lambda x: x[1], reverse=True)[:10]
for pattern, count in sorted_patterns:
    percentage = (count / len(prefixes)) * 100
    print(f'  {count:3d}x ({percentage:5.1f}%): "{pattern}..."')

# Check for generic/vague terms
generic_terms = ['discusses', 'describes', 'explains', 'provides', 'presents', 
                 'contains', 'shows', 'this chunk', 'this document', 'the chunk']

term_counts = {term: sum(1 for p in prefixes if term.lower() in p.lower()) 
               for term in generic_terms}

print(f'\nGeneric Terms Usage:')
for term, count in sorted(term_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / len(prefixes)) * 100
    print(f'  "{term}": {count} times ({percentage:.1f}%)')

# Statistics
print(f'\n{"="*80}')
print("STATISTICS")
print("="*80)

prefix_lengths = [len(p) for p in prefixes]
avg_prefix_len = sum(prefix_lengths) / len(prefix_lengths)
min_prefix_len = min(prefix_lengths)
max_prefix_len = max(prefix_lengths)

print(f'Average prefix length: {avg_prefix_len:.1f} characters')
print(f'Min prefix length: {min_prefix_len} characters')
print(f'Max prefix length: {max_prefix_len} characters')
print(f'Unique starting patterns: {len(starts_with)} / {len(prefixes)}')
print(f'Pattern diversity: {len(starts_with) / len(prefixes) * 100:.1f}%')

# Check specificity
print(f'\n{"="*80}')
print("SPECIFICITY ANALYSIS")
print("="*80)

# Count how many prefixes contain specific technical terms
technical_terms = ['Transformer', 'BERT', 'RAG', 'attention', 'encoder', 'decoder',
                   'embedding', 'layer', 'model', 'neural', 'architecture']

specific_count = sum(1 for p in prefixes 
                     if any(term.lower() in p.lower() for term in technical_terms))

print(f'Prefixes with technical terms: {specific_count} / {len(prefixes)} ({specific_count/len(prefixes)*100:.1f}%)')
print(f'Generic prefixes (no technical terms): {len(prefixes) - specific_count} / {len(prefixes)} ({(len(prefixes)-specific_count)/len(prefixes)*100:.1f}%)')

print(f'\n{"="*80}')
print("DIAGNOSIS")
print("="*80)

print("\nKey Findings:")

# High generic term usage indicates low specificity
avg_generic_usage = sum(term_counts.values()) / len(term_counts)
if avg_generic_usage > len(prefixes) * 0.3:
    print("HIGH generic term usage - prefixes may be too vague")
else:
    print("LOW generic term usage - prefixes appear specific")

# Low pattern diversity indicates repetitive prefixes
if len(starts_with) / len(prefixes) < 0.5:
    print("LOW pattern diversity - many prefixes start similarly")
else:
    print("HIGH pattern diversity - prefixes are varied")

# Low technical term usage indicates lack of domain knowledge
if specific_count / len(prefixes) < 0.4:
    print("LOW technical specificity - missing domain-specific context")
else:
    print("HIGH technical specificity - contains domain knowledge")

print("\nRecommendation:")
if avg_generic_usage > len(prefixes) * 0.3 or len(starts_with) / len(prefixes) < 0.5:
    print("The contextual prefixes appear TOO GENERIC and REPETITIVE.")
    print("This explains why contextual retrieval has 0% recall!")
    print("\nSolution: Improve the enrichment prompt to generate more specific,")
    print("distinctive context that includes document-specific details.")
else:
    print("The prefixes appear reasonably specific. The 0% recall issue may be")
    print("due to other factors (e.g., query-document mismatch, embedding model).")
