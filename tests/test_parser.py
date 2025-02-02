from cot2tot.parser import parse_cot
import json
import pytest
from pathlib import Path

@pytest.fixture
def example_cot():
    """Load test data from JSON file."""
    file_path = Path(__file__).parent / "fixtures" / "llm_output.json"
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def test_json_import(example_cot):
    """Test using the imported JSON data."""
    assert "conversation" in example_cot 
    assert len(example_cot["conversation"]) == 2