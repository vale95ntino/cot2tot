import pytest
from cot2tot.utils import extract_between

@pytest.mark.parametrize(
    "text, key_start, key_end, expected",
    [
        # Basic cases
        ("Hello [start]world[end]!", "[start]", "[end]", ["world"]),
        ("This is (a test) string with (multiple) matches.", "(", ")", ["a test", "multiple"]),
        ("No start or end here.", "[start]", "[end]", []),  # No match
        
        # Edge cases
        ("Edge [start] case[end]", "[start]", "[end]", [" case"]),  # Leading space
        ("[start]Only one match[end]", "[start]", "[end]", ["Only one match"]),  # Exact match
        ("Multiple [key]values[key] in [key]a row[key]", "[key]", "[key]", ["values", "a row"]),  # Same start and end

        # Handling multiple occurrences in a row
        ("Data [start]first[end][start]second[end][start]third[end]", "[start]", "[end]", ["first", "second", "third"]),

        # Special characters in delimiters
        ("Using delimiters {here} and {there}", "{", "}", ["here", "there"]),
        ("Special [*]cases[*] are tricky.", "[*]", "[*]", ["cases"]),

        # If key_start or key_end doesn't exist
        ("Some text with no markers.", "[missing]", "[end]", []),
        ("Missing an end marker [start] here", "[start]", "[end]", []),
    ],
)
def test_extract_between(text, key_start, key_end, expected):
    """Test extract_between with multiple cases."""
    result = extract_between(text, key_start, key_end)
    assert result == expected
