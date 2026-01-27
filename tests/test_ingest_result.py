# tests/test_ingest_result.py
from datetime import datetime
from chinvex.ingest import IngestRunResult

def test_ingest_run_result_creation():
    """Test IngestRunResult can be created with all required fields."""
    result = IngestRunResult(
        run_id="test_run_123",
        context="TestContext",
        started_at=datetime.now(),
        finished_at=datetime.now(),
        new_doc_ids=["doc1", "doc2"],
        updated_doc_ids=["doc3"],
        new_chunk_ids=["chunk1", "chunk2", "chunk3"],
        skipped_doc_ids=["doc4"],
        error_doc_ids=["doc5"],
        stats={"files_scanned": 10, "total_chunks": 25}
    )

    assert result.run_id == "test_run_123"
    assert result.context == "TestContext"
    assert len(result.new_doc_ids) == 2
    assert len(result.new_chunk_ids) == 3
    assert result.stats["files_scanned"] == 10
