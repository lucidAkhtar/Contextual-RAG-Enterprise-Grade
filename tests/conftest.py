"""Test configuration for pytest."""

import pytest
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return {
        "test_pdf": "data/research_paper.pdf",
        "test_ground_truth": "data/ground_truth.json"
    }
