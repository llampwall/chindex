"""Test chunking v2 improvements."""
import pytest
from chinvex.chunking import chunk_with_overlap, chunk_generic_file, find_best_split, chunk_markdown_file, extract_python_boundaries, chunk_python_file


def test_chunk_with_overlap_basic():
    """Test that overlap is applied between chunks."""
    text = "a" * 5000  # 5000 chars
    chunks = chunk_with_overlap(text, size=3000, overlap=300)

    # Expected: 2-3 chunks with overlap
    assert len(chunks) >= 2

    # Check overlap exists between consecutive chunks
    for i in range(len(chunks) - 1):
        start1, end1 = chunks[i]
        start2, end2 = chunks[i + 1]

        # start2 should be before end1 (overlap)
        assert start2 < end1
        # Overlap should be ~300 chars
        overlap_size = end1 - start2
        assert 250 <= overlap_size <= 350


def test_chunk_with_overlap_handles_small_text():
    """Test that small text becomes single chunk."""
    text = "small text"
    chunks = chunk_with_overlap(text, size=3000, overlap=300)
    assert len(chunks) == 1
    assert chunks[0] == (0, len(text))


def test_chunk_generic_file_uses_overlap():
    """Test that generic file chunking uses overlap."""
    text = "Lorem ipsum " * 500  # ~6000 chars
    chunks = chunk_generic_file(text)

    assert len(chunks) >= 2
    # Verify chunks have overlap metadata
    for i in range(len(chunks) - 1):
        chunk1 = chunks[i]
        chunk2 = chunks[i + 1]
        # Check that text overlaps
        text1 = text[chunk1.char_start:chunk1.char_end]
        text2 = text[chunk2.char_start:chunk2.char_end]
        # Last ~300 chars of chunk1 should appear in chunk2
        overlap_text = text1[-300:]
        assert overlap_text in text2


def test_find_best_split_prefers_headers():
    """Test that semantic boundaries are preferred."""
    text = "a" * 2800 + "\n## Header\n" + "b" * 500
    # Target split at 3000, should prefer header at 2801
    split_pos = find_best_split(text, target_pos=3000, size=3000)

    # Should split at or near the header
    assert 2700 <= split_pos <= 2900
    assert text[split_pos:split_pos + 10].startswith("\n## ")


def test_find_best_split_handles_no_boundaries():
    """Test fallback when no semantic boundaries found."""
    text = "a" * 5000  # No boundaries
    split_pos = find_best_split(text, target_pos=3000, size=3000)

    # Should return near target (may find \n or target itself)
    assert 2700 <= split_pos <= 3300


def test_chunk_markdown_respects_boundaries():
    """Test that markdown files chunk at headers."""
    # Create realistic markdown with paragraph breaks
    paragraphs = []
    for i in range(50):
        paragraphs.append(f"This is paragraph {i}. " * 10)  # ~200 chars each

    text = """# Title

Some intro text.

## Section 1

""" + "\n\n".join(paragraphs[:25]) + """

## Section 2

""" + "\n\n".join(paragraphs[25:])

    chunks = chunk_markdown_file(text)

    # Verify chunks start at semantic boundaries
    for chunk in chunks:
        if chunk.ordinal > 0:  # Skip first chunk
            chunk_text = text[chunk.char_start:chunk.char_end]
            # Should start with newline or header
            assert chunk_text.startswith(("\n", "#", "##"))


def test_extract_python_boundaries():
    """Test Python AST boundary extraction."""
    text = '''"""Module docstring."""

def function_one():
    return 1

class MyClass:
    def method(self):
        pass

@decorator
def function_two():
    return 2

if __name__ == "__main__":
    main()
'''

    boundaries = extract_python_boundaries(text)

    # Expected: boundaries at function/class definitions
    assert len(boundaries) >= 3  # function_one, MyClass, function_two, __main__
    # First boundary should be after docstring
    assert boundaries[0] > 20  # After module docstring


def test_chunk_python_file_respects_functions():
    """Test that Python files chunk at function boundaries."""
    # Create Python file with multiple functions
    functions = []
    for i in range(10):
        functions.append(f'''
def function_{i}():
    """Function {i} docstring."""
    # Implementation
    {f"x{i} = {i}; " * 50}
    return {i}
''')

    text = "\n".join(functions)
    chunks = chunk_python_file(text, max_chars=3000)

    # Verify multiple chunks created
    assert len(chunks) >= 2

    # Verify each chunk starts at a function boundary (or start of file)
    for chunk in chunks:
        if chunk.ordinal > 0:
            chunk_text = text[chunk.char_start:chunk.char_end]
            # Should start with 'def ' or at file start
            assert chunk_text.lstrip().startswith("def ") or chunk.char_start == 0


def test_chunk_python_file_handles_syntax_errors():
    """Test that invalid Python falls back to generic."""
    text = "def broken(\n  # Missing closing paren"
    chunks = chunk_python_file(text)

    # Should not crash, should return chunks
    assert len(chunks) >= 1
