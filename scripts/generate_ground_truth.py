"""
Ground truth generation script.
Creates QA pairs from PDF document using LLM assistance.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import re

from config.settings import get_settings
from src.core.document_processor import DocumentProcessor
from src.core.llm_factory import LLMFactory
from src.models.schemas import GroundTruthQA
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class GroundTruthGenerator:
    """
    Generates ground truth QA pairs from PDF documents.
    Uses LLM to create diverse, high-quality questions and answers.
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize generator.
        
        Args:
            pdf_path: Path to PDF document
        """
        self.pdf_path = pdf_path
        self.doc_processor = DocumentProcessor()
        self.llm = LLMFactory.get_llm()
        
        # Load document
        self.documents = self.doc_processor.load_pdf(pdf_path)
        logger.info(f"Loaded {len(self.documents)} pages from PDF")
    
    def generate_qa_pairs(
        self,
        num_pairs: int = 10,
        pages: List[int] = None
    ) -> List[GroundTruthQA]:
        """
        Generate QA pairs from document.
        
        Args:
            num_pairs: Number of QA pairs to generate
            pages: Specific pages to generate from (None for all)
        
        Returns:
            List of ground truth QA pairs
        """
        logger.info(f"Generating {num_pairs} QA pairs...")
        
        qa_pairs = []
        target_pages = pages or [doc.metadata["page"] for doc in self.documents]
        
        # Distribute questions across pages
        pages_to_use = self._select_pages(target_pages, num_pairs)
        
        for page_num in pages_to_use:
            try:
                # Get document for this page
                doc = next(
                    (d for d in self.documents if d.metadata["page"] == page_num),
                    None
                )
                
                if not doc:
                    logger.warning(f"Page {page_num} not found")
                    continue
                
                # Generate QA pair for this page
                qa = self._generate_qa_for_page(doc)
                if qa:
                    qa_pairs.append(qa)
                    logger.info(
                        f"Generated QA pair {len(qa_pairs)}/{num_pairs} from page {page_num}"
                    )
                
                if len(qa_pairs) >= num_pairs:
                    break
                    
            except Exception as e:
                logger.error(f"Error generating QA for page {page_num}: {e}")
                continue
        
        logger.info(f"Successfully generated {len(qa_pairs)} QA pairs")
        return qa_pairs
    
    def _select_pages(self, available_pages: List[int], num_pairs: int) -> List[int]:
        """Select pages to generate questions from."""
        import random
        
        # If we need more pairs than pages, repeat some pages
        if num_pairs <= len(available_pages):
            return random.sample(available_pages, num_pairs)
        else:
            # Sample with replacement
            return random.choices(available_pages, k=num_pairs)
    
    def _generate_qa_for_page(self, document: Any) -> GroundTruthQA:
        """Generate QA pair for a specific page."""
        page_num = document.metadata["page"]
        content = document.get_content()
        
        # Truncate if too long
        max_content_length = 2000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        # Create prompt for QA generation
        prompt = f"""Based on the following text excerpt from page {page_num}, generate ONE high-quality question and answer pair.

Text:
{content}

Requirements:
- The question should be specific and answerable from the text
- The answer should be concise (1-3 sentences) and directly from or inferred from the text
- Focus on key concepts, findings, or important details
- Avoid yes/no questions

Format your response EXACTLY as:
QUESTION: <your question here>
ANSWER: <your answer here>

Do not include any other text or explanation."""
        
        try:
            # Get LLM response
            response = self.llm.complete(prompt)
            response_text = response.text.strip()
            
            # Parse response
            question, answer = self._parse_qa_response(response_text)
            
            if not question or not answer:
                logger.warning(f"Failed to parse QA from page {page_num}")
                return None
            
            # Find answer span in original text (approximate)
            span_start, span_end = self._find_answer_span(content, answer)
            
            # Create QA object
            qa = GroundTruthQA(
                question=question,
                answer=answer,
                page=page_num,
                span_start=span_start,
                span_end=span_end,
                context=content[:500]  # First 500 chars as context
            )
            
            return qa
            
        except Exception as e:
            logger.error(f"Error generating QA for page {page_num}: {e}")
            return None
    
    def _parse_qa_response(self, response: str) -> tuple:
        """Parse question and answer from LLM response."""
        # Try to extract QUESTION: and ANSWER: patterns
        question_match = re.search(r'QUESTION:\s*(.+?)(?=ANSWER:|$)', response, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r'ANSWER:\s*(.+?)$', response, re.DOTALL | re.IGNORECASE)
        
        question = question_match.group(1).strip() if question_match else ""
        answer = answer_match.group(1).strip() if answer_match else ""
        
        # Clean up
        question = question.replace('\n', ' ').strip()
        answer = answer.replace('\n', ' ').strip()
        
        return question, answer
    
    def _find_answer_span(self, text: str, answer: str) -> tuple:
        """
        Find approximate span of answer in text.
        Returns (start, end) positions.
        """
        # Try exact match first
        pos = text.lower().find(answer.lower())
        if pos != -1:
            return (pos, pos + len(answer))
        
        # Try matching first few words
        words = answer.split()[:5]
        search_phrase = ' '.join(words)
        pos = text.lower().find(search_phrase.lower())
        if pos != -1:
            return (pos, pos + len(search_phrase))
        
        # Default to beginning if not found
        return (0, min(100, len(text)))
    
    def save_to_json(self, qa_pairs: List[GroundTruthQA], output_path: str) -> None:
        """
        Save QA pairs to JSON file.
        
        Args:
            qa_pairs: List of QA pairs
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict
        data = [qa.dict() for qa in qa_pairs]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(qa_pairs)} QA pairs to: {output_path}")
    
    def generate_manual_template(self, output_path: str) -> None:
        """
        Generate a template JSON file for manual QA entry.
        Useful for creating high-quality ground truth manually.
        
        Args:
            output_path: Output file path
        """
        template = []
        
        for i, doc in enumerate(self.documents[:10], 1):
            page_num = doc.metadata["page"]
            content_preview = doc.get_content()[:200] + "..."
            
            template.append({
                "question": f"<Enter question {i} about content from page {page_num}>",
                "answer": "<Enter concise answer>",
                "page": page_num,
                "span_start": 0,
                "span_end": 100,
                "context": content_preview,
                "note": "Edit this entry with actual question and answer"
            })
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved template to: {output_path}")


def main():
    """Main execution."""
    settings = get_settings()
    
    logger.info("=== Ground Truth Generation ===")
    
    # Check if PDF exists
    if not Path(settings.pdf_path).exists():
        logger.error(f"PDF file not found: {settings.pdf_path}")
        logger.info("Please place a research paper PDF at: data/research_paper.pdf")
        return
    
    # Initialize generator
    logger.info(f"Processing PDF: {settings.pdf_path}")
    generator = GroundTruthGenerator(settings.pdf_path)
    
    # Generate QA pairs
    qa_pairs = generator.generate_qa_pairs(num_pairs=10)
    
    if qa_pairs:
        # Save to JSON
        generator.save_to_json(qa_pairs, settings.ground_truth_path)
        
        # Print preview
        print("\n" + "="*80)
        print("GENERATED QA PAIRS PREVIEW")
        print("="*80)
        for i, qa in enumerate(qa_pairs[:3], 1):
            print(f"\n{i}. Page {qa.page}")
            print(f"   Q: {qa.question}")
            print(f"   A: {qa.answer}")
        
        if len(qa_pairs) > 3:
            print(f"\n... and {len(qa_pairs) - 3} more pairs")
        
        print("\n" + "="*80)
        print(f"✓ Ground truth saved to: {settings.ground_truth_path}")
        print("="*80)
    else:
        logger.error("No QA pairs were generated")
        
        # Generate template for manual entry
        template_path = settings.ground_truth_path.replace('.json', '_template.json')
        generator.generate_manual_template(template_path)
        logger.info(f"Template generated at: {template_path}")
        logger.info("You can manually fill in the questions and answers")


if __name__ == "__main__":
    main()
