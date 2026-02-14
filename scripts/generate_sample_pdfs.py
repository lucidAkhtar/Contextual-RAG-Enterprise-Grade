"""
Generate sample PDF documents with tables and data for testing.
Creates realistic documents to demonstrate multi-PDF RAG capabilities.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
from pathlib import Path
import json

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_research_paper_pdf(output_path: str):
    """Create a sample research paper PDF with tables."""
    
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    story.append(Paragraph("Contextual Retrieval in Modern RAG Systems", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Abstract
    story.append(Paragraph("Abstract", styles['Heading2']))
    abstract_text = """
    This research explores novel approaches to contextual retrieval-augmented generation (RAG) systems.
    We demonstrate a 23% improvement in retrieval accuracy through semantic understanding and contextual
    enrichment of document chunks. Our hybrid retrieval method combines semantic embeddings, BM25, and
    TF-IDF approaches, showing superior performance across multiple benchmarks.
    """
    story.append(Paragraph(abstract_text, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # Introduction
    story.append(Paragraph("1. Introduction", styles['Heading2']))
    intro_text = """
    Traditional RAG systems suffer from context loss during document chunking. When documents are split
    into smaller chunks for embedding, valuable contextual information from surrounding sections is lost.
    This paper introduces a contextual enrichment approach that preserves document context, significantly
    improving retrieval relevance and answer quality.
    """
    story.append(Paragraph(intro_text, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # Methodology
    story.append(Paragraph("2. Methodology", styles['Heading2']))
    method_text = """
    Our approach consists of three main components: (1) Contextual chunk enrichment using LLM-generated
    context descriptions, (2) Hybrid retrieval combining multiple ranking methods, and (3) Reciprocal
    rank fusion for result aggregation. The system processes documents through a multi-stage pipeline
    that maintains semantic coherence while enabling efficient retrieval.
    """
    story.append(Paragraph(method_text, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # Performance Table
    story.append(Paragraph("3. Experimental Results", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    # Create performance comparison table
    data = [
        ['Method', 'Recall@5', 'Latency (ms)', 'Semantic Similarity'],
        ['Traditional RAG', '0.67', '145', '0.78'],
        ['BM25 Only', '0.71', '95', '0.74'],
        ['Contextual RAG', '0.82', '156', '0.89'],
        ['Hybrid (Proposed)', '0.91', '178', '0.93']
    ]
    
    table = Table(data, colWidths=[2*inch, 1.2*inch, 1.3*inch, 1.8*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a90e2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f0f0')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
    ]))
    
    story.append(table)
    story.append(Spacer(1, 0.2*inch))
    
    results_text = """
    Table 1 presents our benchmark results across four retrieval methods. The proposed hybrid approach
    achieves the highest recall@5 (0.91) and semantic similarity (0.93), demonstrating superior
    retrieval quality. While latency increases slightly to 178ms, the quality gains justify this
    trade-off for most applications.
    """
    story.append(Paragraph(results_text, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # Dataset section
    story.append(Paragraph("4. Datasets and Evaluation", styles['Heading2']))
    dataset_text = """
    We evaluated our system on three benchmark datasets: MS MARCO for passage retrieval, Natural
    Questions for open-domain QA, and a custom technical documentation corpus. Each dataset presents
    unique challenges in terms of document length, query complexity, and domain specificity. Our hybrid
    approach consistently outperforms baseline methods across all datasets.
    """
    story.append(Paragraph(dataset_text, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # Limitations
    story.append(Paragraph("5. Limitations and Future Work", styles['Heading2']))
    limitations_text = """
    The primary limitation of our approach is the increased computational cost during indexing due to
    LLM-based contextual enrichment. Each chunk requires an LLM call to generate contextual descriptions,
    adding approximately 200-300ms per chunk. Future work will explore caching strategies and batch
    processing to mitigate this overhead. Additionally, very short documents (< 500 tokens) show
    diminished benefits from contextual enrichment.
    """
    story.append(Paragraph(limitations_text, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # Conclusion
    story.append(Paragraph("6. Conclusion", styles['Heading2']))
    conclusion_text = """
    This paper demonstrates that contextual enrichment significantly improves RAG system performance.
    By preserving document context during chunking and employing hybrid retrieval methods, we achieve
    23% better recall and 19% higher semantic similarity compared to traditional approaches. These
    improvements translate to more accurate and relevant answers in production deployments.
    """
    story.append(Paragraph(conclusion_text, styles['BodyText']))
    
    # Build PDF
    doc.build(story)
    logger.info(f"Created research paper PDF: {output_path}")


def create_financial_data_pdf(output_path: str):
    """Create a financial report PDF with data tables."""
    
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=20,
        alignment=1
    )
    
    story.append(Paragraph("Annual Financial Report 2025", title_style))
    story.append(Paragraph("TechVentures Inc.", styles['Heading3']))
    story.append(Spacer(1, 0.4*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    summary_text = """
    TechVentures Inc. achieved record revenue of $450 million in fiscal year 2025, representing 28%
    year-over-year growth. Our AI and cloud services divisions drove this growth, with operating margins
    improving to 32%. Strong customer retention and successful product launches position us well for
    continued expansion in 2026.
    """
    story.append(Paragraph(summary_text, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # Revenue Table
    story.append(Paragraph("Quarterly Revenue Breakdown", styles['Heading3']))
    story.append(Spacer(1, 0.1*inch))
    
    revenue_data = [
        ['Quarter', 'Revenue ($M)', 'YoY Growth', 'Operating Margin'],
        ['Q1 2025', '105', '22%', '29%'],
        ['Q2 2025', '112', '25%', '31%'],
        ['Q3 2025', '118', '30%', '33%'],
        ['Q4 2025', '115', '28%', '34%'],
        ['Total 2025', '450', '28%', '32%']
    ]
    
    revenue_table = Table(revenue_data, colWidths=[1.5*inch, 1.5*inch, 1.3*inch, 1.6*inch])
    revenue_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#d5f4e6')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor('#f9f9f9')])
    ]))
    
    story.append(revenue_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Product Performance
    story.append(Paragraph("Product Division Performance", styles['Heading3']))
    story.append(Spacer(1, 0.1*inch))
    
    product_data = [
        ['Division', 'Revenue ($M)', '% of Total', 'Growth Rate'],
        ['AI Services', '180', '40%', '45%'],
        ['Cloud Platform', '135', '30%', '25%'],
        ['Enterprise Software', '90', '20%', '15%'],
        ['Consulting', '45', '10%', '12%']
    ]
    
    product_table = Table(product_data, colWidths=[2*inch, 1.5*inch, 1.2*inch, 1.3*inch])
    product_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f8ff')])
    ]))
    
    story.append(product_table)
    story.append(Spacer(1, 0.2*inch))
    
    analysis_text = """
    Our AI Services division continues to be the primary growth driver, capturing 40% of total revenue
    with 45% year-over-year growth. The success of our GPT-powered analytics platform and automated
    workflow tools has resonated strongly with enterprise customers. Cloud Platform revenue grew 25%,
    supported by increased adoption of our managed Kubernetes services.
    """
    story.append(Paragraph(analysis_text, styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))
    
    # Key Metrics
    story.append(Paragraph("Key Performance Indicators", styles['Heading3']))
    story.append(Spacer(1, 0.1*inch))
    
    metrics_data = [
        ['Metric', '2024', '2025', 'Change'],
        ['Active Customers', '12,500', '16,200', '+30%'],
        ['Annual Recurring Revenue', '$385M', '$495M', '+29%'],
        ['Customer Retention Rate', '89%', '92%', '+3pp'],
        ['Net Promoter Score', '67', '73', '+6pts'],
        ['Employee Count', '2,800', '3,400', '+21%']
    ]
    
    metrics_table = Table(metrics_data, colWidths=[2.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e67e22')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fef5e7')])
    ]))
    
    story.append(metrics_table)
    story.append(Spacer(1, 0.2*inch))
    
    metrics_text = """
    Customer growth accelerated to 30%, with strong traction in mid-market enterprises. Our 92% retention
    rate reflects high customer satisfaction and the sticky nature of our platform. We expanded our team
    by 21% to support growth, with key hires in engineering, sales, and customer success roles.
    """
    story.append(Paragraph(metrics_text, styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))
    
    # Outlook
    story.append(Paragraph("2026 Outlook", styles['Heading2']))
    outlook_text = """
    We project 2026 revenue of $575-600 million, representing 28-33% growth. Key drivers include
    expansion of our AI Services portfolio, geographic expansion into APAC markets, and new product
    launches scheduled for Q2. We will continue investing in R&D while maintaining operating margins
    above 30% through operational efficiency improvements.
    """
    story.append(Paragraph(outlook_text, styles['BodyText']))
    
    # Build PDF
    doc.build(story)
    logger.info(f"Created financial report PDF: {output_path}")


def create_employee_handbook_pdf(output_path: str):
    """Create an employee handbook PDF with policy tables."""
    
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#8e44ad'),
        spaceAfter=20,
        alignment=1
    )
    
    story.append(Paragraph("Employee Handbook", title_style))
    story.append(Paragraph("TechVentures Inc. - 2026 Edition", styles['Heading3']))
    story.append(Spacer(1, 0.3*inch))
    
    # Welcome
    story.append(Paragraph("Welcome to TechVentures", styles['Heading2']))
    welcome_text = """
    Welcome to TechVentures Inc.! This handbook provides essential information about our policies,
    benefits, and workplace culture. We're committed to creating an inclusive, innovative environment
    where every employee can thrive. Please review this handbook carefully and reach out to HR with
    any questions.
    """
    story.append(Paragraph(welcome_text, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # PTO Policy
    story.append(Paragraph("Paid Time Off (PTO) Policy", styles['Heading3']))
    story.append(Spacer(1, 0.1*inch))
    
    pto_data = [
        ['Employee Level', 'Annual PTO Days', 'Sick Days', 'Holidays'],
        ['Entry Level (0-2 years)', '15', '10', '12'],
        ['Mid Level (3-5 years)', '20', '12', '12'],
        ['Senior (6-10 years)', '25', '15', '12'],
        ['Executive (10+ years)', '30', 'Unlimited', '12']
    ]
    
    pto_table = Table(pto_data, colWidths=[2*inch, 1.5*inch, 1.2*inch, 1.2*inch])
    pto_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f4ecf7')])
    ]))
    
    story.append(pto_table)
    story.append(Spacer(1, 0.2*inch))
    
    pto_text = """
    PTO accrues monthly and can be used for vacation, personal matters, or any purpose. Unused PTO
    rolls over up to 10 days per year. We encourage employees to take time off to recharge and maintain
    work-life balance. Request PTO through our HR portal at least 2 weeks in advance for planned absences.
    """
    story.append(Paragraph(pto_text, styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))
    
    # Benefits
    story.append(Paragraph("Health & Wellness Benefits", styles['Heading3']))
    story.append(Spacer(1, 0.1*inch))
    
    benefits_data = [
        ['Benefit Type', 'Company Coverage', 'Employee Cost', 'Details'],
        ['Health Insurance', '85%', '15%', 'PPO & HMO options'],
        ['Dental Insurance', '75%', '25%', 'Full coverage'],
        ['Vision Insurance', '100%', '$0', 'Annual eye exam'],
        ['Life Insurance', '100%', '$0', '2x annual salary'],
        ['Gym Membership', '50%', '50%', 'Up to $100/month']
    ]
    
    benefits_table = Table(benefits_data, colWidths=[1.8*inch, 1.5*inch, 1.3*inch, 1.8*inch])
    benefits_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16a085')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#d5f4e6')])
    ]))
    
    story.append(benefits_table)
    story.append(Spacer(1, 0.2*inch))
    
    benefits_text = """
    Our comprehensive benefits package supports your health and financial well-being. Benefits become
    effective on your first day of employment. Enroll during your first week or during the annual open
    enrollment period in November. We also offer 401(k) matching up to 6% of salary.
    """
    story.append(Paragraph(benefits_text, styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))
    
    # Remote Work
    story.append(Paragraph("Remote Work & Hybrid Policy", styles['Heading3']))
    remote_text = """
    TechVentures embraces flexible work arrangements. Employees can work remotely up to 3 days per week,
    with Tuesday and Thursday designated as in-office collaboration days. Full remote work is available
    for roles that don't require physical presence, subject to manager approval. All employees receive
    a $500 annual stipend for home office equipment.
    """
    story.append(Paragraph(remote_text, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # Professional Development
    story.append(Paragraph("Professional Development", styles['Heading3']))
    dev_text = """
    We invest $2,000 per employee annually for professional development, including conferences, courses,
    and certifications. Career growth conversations happen quarterly with your manager. We also offer
    internal mentorship programs and leadership training for high-potential employees.
    """
    story.append(Paragraph(dev_text, styles['BodyText']))
    
    # Build PDF
    doc.build(story)
    logger.info(f"Created employee handbook PDF: {output_path}")


def main():
    """Generate all sample PDFs."""
    logger.info("Starting PDF generation...")
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Generate PDFs
        create_research_paper_pdf("data/research_paper.pdf")
        create_financial_data_pdf("data/financial_report.pdf")
        create_employee_handbook_pdf("data/employee_handbook.pdf")
        
        logger.info("All PDFs generated successfully!")
        print("\n✅ Generated 3 sample PDFs:")
        print("   1. data/research_paper.pdf - Academic paper with performance tables")
        print("   2. data/financial_report.pdf - Financial report with revenue/metrics tables")
        print("   3. data/employee_handbook.pdf - HR handbook with policy tables")
        
    except Exception as e:
        logger.error(f"Error generating PDFs: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("\nMake sure 'reportlab' is installed: pip install reportlab")


if __name__ == "__main__":
    main()
