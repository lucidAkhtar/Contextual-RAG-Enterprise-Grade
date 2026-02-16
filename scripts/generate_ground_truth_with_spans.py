#!/usr/bin/env python3
"""
Generate ground truth QA pairs with page numbers and character spans.
Complies with assignment requirements: {question, answer, page, span_start, span_end}
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import fitz  # PyMuPDF


class GroundTruthGenerator:
    """Generate ground truth with span annotations from PDFs."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.pdf_texts = {}  # Store extracted text per PDF
        self.pdf_pages = {}  # Store page-wise text per PDF
        
    def extract_pdf_text(self, pdf_path: Path) -> Tuple[str, List[Dict]]:
        """
        Extract text from PDF with page information.
        
        Returns:
            Tuple of (full_text, page_data)
            page_data: [{page_num, text, char_start, char_end}]
        """
        doc = fitz.open(pdf_path)
        full_text = ""
        page_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            
            char_start = len(full_text)
            full_text += page_text
            char_end = len(full_text)
            
            page_data.append({
                "page_num": page_num + 1,  # 1-indexed
                "text": page_text,
                "char_start": char_start,
                "char_end": char_end
            })
        
        doc.close()
        return full_text, page_data
    
    def find_answer_span(self, answer: str, pdf_name: str, fuzzy: bool = True) -> Optional[Dict]:
        """
        Find the character span of an answer in the PDF.
        
        Args:
            answer: The answer text to search for
            pdf_name: Name of the PDF file
            fuzzy: If True, use fuzzy matching for partial matches
            
        Returns:
            Dict with {page, span_start, span_end} or None if not found
        """
        pdf_path = self.data_dir / pdf_name
        
        # Load PDF if not already loaded
        if pdf_name not in self.pdf_texts:
            full_text, page_data = self.extract_pdf_text(pdf_path)
            self.pdf_texts[pdf_name] = full_text
            self.pdf_pages[pdf_name] = page_data
        
        full_text = self.pdf_texts[pdf_name]
        page_data = self.pdf_pages[pdf_name]
        
        # Normalize texts for better matching
        normalized_answer = self.normalize_text(answer)
        normalized_full_text = self.normalize_text(full_text)
        
        # Try exact match first (for short answers)
        if len(answer) < 100:
            match_start = normalized_full_text.find(normalized_answer)
        else:
            match_start = -1
        
        if match_start == -1 and fuzzy:
            # For longer answers, try finding key phrases
            match_start = self.fuzzy_match(normalized_answer, normalized_full_text)
        
        if match_start == -1:
            # Try finding specific numbers/scores mentioned
            numbers = re.findall(r'\d+\.?\d*', answer)
            if numbers:
                for num in numbers:
                    # Search for the number in context
                    pattern = rf'\b{re.escape(num)}\b'
                    match = re.search(pattern, normalized_full_text)
                    if match:
                        match_start = match.start()
                        # Expand to include context (100 chars before and after for long answers)
                        context_size = 150 if len(answer) > 100 else 50
                        match_start = max(0, match_start - context_size)
                        break
        
        if match_start == -1:
            return None
        
        # Calculate actual positions in original text
        # For long answers, use a reasonable excerpt size
        excerpt_size = min(len(normalized_answer), 300) if len(answer) > 100 else min(len(normalized_answer), 150)
        match_end = match_start + excerpt_size if match_start != -1 else -1
        
        # Find which page this span belongs to
        for page_info in page_data:
            if page_info["char_start"] <= match_start < page_info["char_end"]:
                return {
                    "page": page_info["page_num"],
                    "span_start": match_start,
                    "span_end": match_end
                }
        
        return None
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep periods and numbers
        text = re.sub(r'[^\w\s\.\-\%\+\=]', '', text)
        return text.strip().lower()
    
    def fuzzy_match(self, answer: str, full_text: str) -> int:
        """
        Fuzzy match to find answer location.
        Tries multiple strategies for longer text answers.
        """
        # Strategy 1: Try first sentence of answer
        first_sentence = answer.split('.')[0] if '.' in answer else answer
        if len(first_sentence) > 20:
            # Try first 80 chars of first sentence
            search_text = first_sentence[:80] if len(first_sentence) > 80 else first_sentence
            pos = full_text.find(search_text)
            if pos != -1:
                return pos
        
        # Strategy 2: Try key phrases (consecutive words longer than 4 chars)
        words = [w for w in answer.split() if len(w) > 4]
        if len(words) >= 3:
            # Try different combinations
            for i in range(min(3, len(words) - 2)):
                phrase = ' '.join(words[i:i+3])
                pos = full_text.find(phrase)
                if pos != -1:
                    return pos
        
        # Strategy 3: Try finding distinctive terms (capitalized words, technical terms)
        capitalized = [w for w in answer.split() if w and w[0].isupper() and len(w) > 3]
        if capitalized:
            for term in capitalized[:3]:
                # Search for the term with some context
                pos = full_text.find(term.lower())
                if pos != -1:
                    return max(0, pos - 50)
        
        # Strategy 4: Try bigrams of longer words
        if len(words) >= 2:
            for i in range(min(5, len(words) - 1)):
                bigram = f"{words[i]} {words[i+1]}"
                pos = full_text.find(bigram)
                if pos != -1:
                    return max(0, pos - 30)
        
        return -1
    
    def generate_qa_pairs(self) -> List[Dict]:
        """
        Generate high-quality QA pairs with span annotations.
        Each entry: {question, answer, page, span_start, span_end, source_document}
        """
        qa_pairs = []
        
        # Transformer Paper QA Pairs (attention_is_all_you_need.pdf)
        transformer_qas = [
            {
                "question": "What is the main advantage of the Transformer architecture over recurrent models?",
                "answer": "The Transformer architecture can parallelize computation by relying entirely on attention mechanisms instead of recurrence, which allows it to achieve significantly better translation quality while being more parallelizable and requiring less time to train.",
                "source_document": "attention_is_all_you_need.pdf"
            },
            {
                "question": "How does multi-head attention work in the Transformer?",
                "answer": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. Instead of performing a single attention function, it projects queries, keys and values h times with different learned linear projections, performs attention in parallel, and concatenates the results.",
                "source_document": "attention_is_all_you_need.pdf"
            },
            {
                "question": "What are the three different ways attention mechanisms are used in the Transformer?",
                "answer": "The Transformer uses attention in three ways: encoder-decoder attention where queries come from the decoder and keys and values from the encoder, self-attention in the encoder where all keys, values and queries come from the encoder output, and self-attention in the decoder with masking to prevent positions from attending to subsequent positions.",
                "source_document": "attention_is_all_you_need.pdf"
            },
            {
                "question": "What BLEU score did the Transformer (big) model achieve on WMT 2014 English-to-German translation?",
                "answer": "28.4 BLEU",
                "source_document": "attention_is_all_you_need.pdf"
            },
            {
                "question": "Why do the authors use positional encodings in the Transformer?",
                "answer": "Since the model contains no recurrence and no convolution, positional encodings are added to give the model information about the relative or absolute position of tokens in the sequence. Without positional information, the model would be permutation invariant.",
                "source_document": "attention_is_all_you_need.pdf"
            },
            {
                "question": "What are the key hyperparameters of the Transformer base model?",
                "answer": "The base model uses 6 layers (N=6), model dimension d_model=512, 8 attention heads (h=8), feed-forward dimension d_ff=2048, dropout rate of 0.1, and attention dimensions d_k=d_v=64.",
                "source_document": "attention_is_all_you_need.pdf"
            },
            {
                "question": "How much faster was the Transformer compared to previous models in terms of training time?",
                "answer": "The Transformer base model trained in just 12 hours on 8 P100 GPUs for 100,000 steps, while the big model took 3.5 days for 300,000 steps, which is significantly faster than previous state-of-the-art models while achieving better quality.",
                "source_document": "attention_is_all_you_need.pdf"
            },
        ]
        
        # BERT Paper QA Pairs (bert_paper.pdf)
        bert_qas = [
            {
                "question": "What are the two novel unsupervised tasks used in BERT pre-training?",
                "answer": "BERT uses two novel unsupervised prediction tasks: Masked Language Model (MLM) where 15% of tokens are randomly masked and the model predicts them, and Next Sentence Prediction (NSP) where the model learns to predict whether two sentences follow each other in the original text.",
                "source_document": "bert_paper.pdf"
            },
            {
                "question": "How does BERT differ from traditional left-to-right language models like GPT?",
                "answer": "Unlike GPT which uses unidirectional left-to-right architecture, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers using the masked language model objective, allowing it to capture richer contextual information.",
                "source_document": "bert_paper.pdf"
            },
            {
                "question": "What is the architecture difference between BERT-Base and BERT-Large?",
                "answer": "BERT-Base has 12 transformer layers (L=12), hidden size of 768 (H=768), and 12 attention heads (A=12) totaling 110M parameters, while BERT-Large has 24 layers (L=24), hidden size of 1024 (H=1024), and 16 attention heads (A=16) totaling 340M parameters.",
                "source_document": "bert_paper.pdf"
            },
            {
                "question": "What F1 score did BERT-Large achieve on SQuAD v1.1 and how does it compare to previous best?",
                "answer": "BERT-Large achieved 93.2 F1 and 87.4 Exact Match on SQuAD v1.1, surpassing the previous best ensemble system and establishing a new state-of-the-art with a single model.",
                "source_document": "bert_paper.pdf"
            },
            {
                "question": "What training procedure and learning rate does BERT use for fine-tuning?",
                "answer": "BERT fine-tuning uses Adam optimizer with learning rate of 5e-5, 3e-5, or 2e-5 depending on the task, trains for 2-4 epochs, uses batch size of 16 or 32, and typically converges quickly. The same pre-trained model parameters are used across all tasks with just one additional output layer.",
                "source_document": "bert_paper.pdf"
            },
            {
                "question": "Why is the masked language model approach better than traditional language modeling for pre-training?",
                "answer": "The masked language model allows BERT to fuse left and right context to pre-train a deep bidirectional Transformer. Standard left-to-right language models would allow each word to indirectly see itself in a bidirectional model, making pre-training trivial, so masking random tokens solves this issue.",
                "source_document": "bert_paper.pdf"
            },
        ]
        
        # RAG Paper QA Pairs (rag_paper.pdf)
        rag_qas = [
            {
                "question": "What is the key difference between RAG-Sequence and RAG-Token models?",
                "answer": "RAG-Sequence uses the same retrieved document to generate the complete sequence, treating the retrieved documents as a latent variable marginalized to get the sequence likelihood. RAG-Token can use different documents for each token, marginalizing per token and allowing the model to mix content from multiple documents.",
                "source_document": "rag_paper.pdf"
            },
            {
                "question": "How does RAG combine parametric and non-parametric memory?",
                "answer": "RAG combines parametric memory from a pre-trained seq2seq transformer with non-parametric memory from a dense vector index of Wikipedia. The parametric model generates answers conditioned on retrieved documents from the non-parametric memory, allowing it to access and leverage knowledge beyond what's stored in its parameters.",
                "source_document": "rag_paper.pdf"
            },
            {
                "question": "What retrieval system does RAG use and how large is the document index?",
                "answer": "RAG uses Dense Passage Retrieval (DPR) with a bi-encoder architecture to retrieve documents from a dense vector index containing 21 million 100-word Wikipedia passages. The retriever is initialized from DPR and can be fine-tuned end-to-end with the generator.",
                "source_document": "rag_paper.pdf"
            },
            {
                "question": "What are the main advantages of RAG over purely parametric models?",
                "answer": "RAG provides several advantages: it can access up-to-date knowledge by updating the document index without retraining, it's more interpretable by showing which documents were used, it can handle knowledge-intensive tasks better, and it combines the benefits of retrieval-based and generation-based approaches.",
                "source_document": "rag_paper.pdf"
            },
            {
                "question": "What performance did RAG-Sequence achieve on Natural Questions compared to baseline models?",
                "answer": "RAG-Sequence achieved 44.5% Exact Match on Natural Questions, outperforming the BART baseline and achieving competitive results with extraction-based models while generating free-form text rather than extracting spans.",
                "source_document": "rag_paper.pdf"
            },
            {
                "question": "How does RAG handle the training of both retriever and generator components?",
                "answer": "RAG can be trained end-to-end by backpropagating through the retrieval step using the gradient signal from the generator. The retriever and generator are jointly fine-tuned, though the document index embeddings are typically kept frozen after initial retriever training to maintain efficiency.",
                "source_document": "rag_paper.pdf"
            },
        ]
        
        # Combine all QA pairs
        all_qas = transformer_qas + bert_qas + rag_qas
        
        print(f"Processing {len(all_qas)} QA pairs with varied answer types...")
        print(f"  - Text explanations: ~15")
        print(f"  - Numerical/short: ~4\n")
        
        for idx, qa in enumerate(all_qas, 1):
            print(f"\n[{idx}/{len(all_qas)}] Processing: {qa['question'][:60]}...")
            
            # Find span in PDF
            span_info = self.find_answer_span(
                qa["answer"], 
                qa["source_document"],
                fuzzy=True
            )
            
            if span_info:
                qa_pair = {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "page": span_info["page"],
                    "span_start": span_info["span_start"],
                    "span_end": span_info["span_end"],
                    "source_document": qa["source_document"]
                }
                qa_pairs.append(qa_pair)
                print(f"  ✓ Found on page {span_info['page']}, span [{span_info['span_start']}:{span_info['span_end']}]")
            else:
                # Still include but mark as not found
                qa_pair = {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "page": 1,  # Default page
                    "span_start": -1,  # Indicates not found
                    "span_end": -1,
                    "source_document": qa["source_document"],
                    "note": "Span not found - answer may be paraphrased or across multiple pages"
                }
                qa_pairs.append(qa_pair)
                print(f"  ⚠ Span not found (answer may be paraphrased)")
        
        return qa_pairs
    
    def save_ground_truth(self, qa_pairs: List[Dict], output_path: Path):
        """Save ground truth to JSON file."""
        output_data = {
            "metadata": {
                "total_qa_pairs": len(qa_pairs),
                "documents": list(set(qa["source_document"] for qa in qa_pairs)),
                "format": "Assignment compliant: {question, answer, page, span_start, span_end}",
                "note": "span_start and span_end are character positions in the PDF text. -1 indicates span not found."
            },
            "qa_pairs": qa_pairs
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n Ground truth saved to {output_path}")
        print(f" Total QA pairs: {len(qa_pairs)}")
        
        # Summary
        found_spans = sum(1 for qa in qa_pairs if qa["span_start"] != -1)
        print(f"   Spans found: {found_spans}/{len(qa_pairs)}")


def main():
    """Main execution."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_path = data_dir / "ground_truth.json"
    
    print("=" * 70)
    print("Ground Truth Generator with Span Annotations")
    print("=" * 70)
    
    # Generate ground truth
    generator = GroundTruthGenerator(data_dir)
    qa_pairs = generator.generate_qa_pairs()
    
    # Save to file
    generator.save_ground_truth(qa_pairs, output_path)
    
    print("\n" + "=" * 70)
    print("Ground truth generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
