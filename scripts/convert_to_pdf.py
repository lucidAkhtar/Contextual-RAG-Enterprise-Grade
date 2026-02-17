#!/usr/bin/env python3
"""
High-quality Markdown to PDF converter with professional styling.
Generates publication-ready PDFs from markdown files.
"""

import sys
import markdown
from pathlib import Path
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

# Professional CSS styling for academic/technical documents
PROFESSIONAL_CSS = """
@page {
    size: A4;
    margin: 1.5cm 1.5cm;
    
    @top-left {
        content: "Contextual RAG System";
        font-size: 8pt;
        color: #666;
    }
    
    @top-right {
        content: "Page " counter(page);
        font-size: 8pt;
        color: #666;
    }
    
    @bottom-center {
        content: "© 2026 | RAG Pipeline Implementation";
        font-size: 7pt;
        color: #999;
    }
}

body {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
    max-width: 100%;
    margin: 0 auto;
    background: white;
}

h1 {
    font-size: 20pt;
    font-weight: bold;
    color: #1a1a1a;
    margin-top: 0;
    margin-bottom: 0.4em;
    padding-bottom: 0.2em;
    border-bottom: 3px solid #2563eb;
    page-break-after: avoid;
}

h2 {
    font-size: 16pt;
    font-weight: bold;
    color: #2563eb;
    margin-top: 1em;
    margin-bottom: 0.4em;
    page-break-after: avoid;
}

h3 {
    font-size: 13pt;
    font-weight: bold;
    color: #1e40af;
    margin-top: 0.8em;
    margin-bottom: 0.3em;
    page-break-after: avoid;
}

h4 {
    font-size: 11pt;
    font-weight: bold;
    color: #3730a3;
    margin-top: 0.6em;
    margin-bottom: 0.2em;
}

/* Paragraphs */
p {
    margin: 0.3em 0;
    text-align: justify;
}

/* Lists */
ul, ol {
    margin: 0.3em 0;
    padding-left: 1.2em;
}

li {
    margin: 0.2em 0;
}

/* Code blocks */
code {
    font-family: 'Monaco', 'Courier New', monospace;
    font-size: 8pt;
    background-color: #f5f5f5;
    padding: 2px 4px;
    border-radius: 3px;
    color: #d73a49;
}

pre {
    background-color: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 3px;
    padding: 8px;
    overflow-x: visible;
    margin: 0.5em 0;
    page-break-inside: auto;
    max-width: 100%;
}

pre code {
    background-color: transparent;
    padding: 0;
    color: #24292e;
    font-size: 6.5pt;
    line-height: 1.2;
    white-space: pre-wrap;
    word-wrap: break-word;
    display: block;
}

/* Tables */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.5em 0;
    font-size: 9pt;
    page-break-inside: auto;
}

thead {
    background-color: #2563eb;
    color: white;
    font-weight: bold;
}

th {
    padding: 6px 8px;
    text-align: left;
    border: 1px solid #ddd;
}

td {
    padding: 5px 8px;
    border: 1px solid #ddd;
}

tbody tr:nth-child(even) {
    background-color: #f9fafb;
}

tbody tr:hover {
    background-color: #f3f4f6;
}

/* Blockquotes */
blockquote {
    border-left: 4px solid #2563eb;
    padding-left: 0.8em;
    margin: 0.5em 0;
    color: #555;
    font-style: italic;
    background-color: #f9fafb;
    padding: 0.3em 0.8em;
}

/* Horizontal rules */
hr {
    border: none;
    border-top: 2px solid #e5e7eb;
    margin: 1em 0;
}

/* Links */
a {
    color: #2563eb;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Emphasis */
strong {
    font-weight: bold;
    color: #1a1a1a;
}

em {
    font-style: italic;
}

/* Special elements */
.checkmark::before {
    content: "✓ ";
    color: #10b981;
    font-weight: bold;
}

.cross::before {
    content: "✗ ";
    color: #ef4444;
    font-weight: bold;
}

/* Icons in text (emoji-like) */
.icon-check {
    color: #10b981;
}

.icon-cross {
    color: #ef4444;
}

.icon-warning {
    color: #f59e0b;
}

/* Highlight boxes */
.highlight {
    background-color: #fef3c7;
    border-left: 4px solid #f59e0b;
    padding: 0.5em 1em;
    margin: 1em 0;
}

/* Page breaks */
.page-break {
    page-break-after: always;
}

/* Table of contents styling */
.toc {
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
    padding: 1em;
    margin: 1em 0 2em 0;
    border-radius: 5px;
}

.toc ul {
    list-style-type: none;
    padding-left: 0;
}

.toc li {
    margin: 0.3em 0;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 9pt;
    font-weight: bold;
    margin: 0 3px;
}

.badge-success {
    background-color: #d1fae5;
    color: #065f46;
}

.badge-info {
    background-color: #dbeafe;
    color: #1e40af;
}

.badge-warning {
    background-color: #fef3c7;
    color: #92400e;
}
"""


def markdown_to_html(markdown_file: Path) -> str:
    """Convert markdown file to HTML with extensions."""
    with open(markdown_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Configure markdown with extensions
    md = markdown.Markdown(
        extensions=[
            'extra',           # Tables, code blocks, etc.
            'codehilite',      # Syntax highlighting
            'toc',             # Table of contents
            'nl2br',           # Newline to break
            'sane_lists',      # Better list handling
            'smarty',          # Smart quotes and dashes
            'pymdownx.emoji',  # Emoji support
            'pymdownx.superfences',  # Advanced code blocks
            'pymdownx.tasklist',  # Task lists
        ],
        extension_configs={
            'codehilite': {
                'guess_lang': True,
                'linenums': False,
            },
            'pymdownx.tasklist': {
                'custom_checkbox': True,
            }
        }
    )
    
    html_content = md.convert(md_content)
    
    # Wrap in complete HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{markdown_file.stem}</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    return full_html


def html_to_pdf(html_content: str, output_pdf: Path) -> None:
    """Convert HTML to high-quality PDF with professional styling."""
    font_config = FontConfiguration()
    
    # Create CSS object
    css = CSS(string=PROFESSIONAL_CSS, font_config=font_config)
    
    # Generate PDF
    HTML(string=html_content).write_pdf(
        output_pdf,
        stylesheets=[css],
        font_config=font_config
    )


def convert_markdown_to_pdf(
    markdown_file: Path,
    output_pdf: Path = None
) -> Path:
    """
    Main conversion function: Markdown -> HTML -> PDF
    
    Args:
        markdown_file: Path to input markdown file
        output_pdf: Path to output PDF file (optional)
    
    Returns:
        Path to generated PDF file
    """
    if not markdown_file.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_file}")
    
    if output_pdf is None:
        output_pdf = markdown_file.with_suffix('.pdf')
    
    print(f"Converting: {markdown_file.name}")
    print(f"Output: {output_pdf.name}")
    
    # Step 1: Markdown -> HTML
    print("   Parsing markdown...")
    html_content = markdown_to_html(markdown_file)
    
    # Step 2: HTML -> PDF
    print("  Applying professional styling...")
    html_to_pdf(html_content, output_pdf)
    
    # Get file size
    size_kb = output_pdf.stat().st_size / 1024
    print(f"  PDF generated: {size_kb:.1f} KB")
    
    return output_pdf


def main():
    """CLI interface for PDF conversion."""
    if len(sys.argv) < 2:
        print("Usage: python convert_to_pdf.py <markdown_file> [output_pdf]")
        print("\nExample:")
        print("  python convert_to_pdf.py SYSTEM_DESIGN.md")
        print("  python convert_to_pdf.py SYSTEM_DESIGN.md output.pdf")
        sys.exit(1)
    
    markdown_file = Path(sys.argv[1])
    output_pdf = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    try:
        result_pdf = convert_markdown_to_pdf(markdown_file, output_pdf)
        print(f"\nSuccess! PDF saved to: {result_pdf}")
    except Exception as e:
        print(f"\n Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
