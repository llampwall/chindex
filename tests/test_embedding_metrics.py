import pytest
from unittest.mock import Mock, patch
from prometheus_client import REGISTRY


def test_embedding_metrics_tracked_ollama():
    """Test that Ollama embedding calls are tracked in metrics."""
    from chinvex.embedding_providers import OllamaProvider

    # Get initial counter value
    before_count = _get_counter_value("chinvex_embeddings_total", {"provider": "ollama"})

    # Create provider and mock the underlying embedder
    provider = OllamaProvider("http://localhost:11434", "mxbai-embed-large")

    with patch("chinvex.embed.OllamaEmbedder") as mock_embedder_class:
        mock_instance = Mock()
        mock_instance.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_embedder_class.return_value = mock_instance

        # Call embed
        provider.embed(["test"])

    # Check metrics were incremented
    after_count = _get_counter_value("chinvex_embeddings_total", {"provider": "ollama"})
    assert after_count == before_count + 1


def test_embedding_metrics_tracked_openai():
    """Test that OpenAI embedding calls are tracked in metrics."""
    from chinvex.embedding_providers import OpenAIProvider

    # Get initial counter value
    before_count = _get_counter_value("chinvex_embeddings_total", {"provider": "openai"})

    # Create provider with mock client
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        provider = OpenAIProvider(api_key="test-key", model="text-embedding-3-small")

    # Mock the client
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
    provider.client.embeddings.create = Mock(return_value=mock_response)

    # Call embed
    provider.embed(["test"])

    # Check metrics were incremented
    after_count = _get_counter_value("chinvex_embeddings_total", {"provider": "openai"})
    assert after_count == before_count + 1


def test_embedding_retries_tracked_openai():
    """Test that OpenAI retry attempts are tracked."""
    from chinvex.embedding_providers import OpenAIProvider
    from openai import RateLimitError

    # Get initial retry counter value
    before_count = _get_counter_value("chinvex_embeddings_retries_total", {"provider": "openai"})

    # Create provider
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        provider = OpenAIProvider(api_key="test-key", model="text-embedding-3-small")

    # Mock the client to fail once then succeed
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]

    call_count = 0
    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RateLimitError("Rate limit", response=Mock(status_code=429), body=None)
        return mock_response

    provider.client.embeddings.create = Mock(side_effect=side_effect)

    # Mock time.sleep to speed up test
    with patch("time.sleep"):
        # Call embed
        provider.embed(["test"])

    # Check retry metrics were incremented
    after_count = _get_counter_value("chinvex_embeddings_retries_total", {"provider": "openai"})
    assert after_count == before_count + 1


def test_digest_generated_metric():
    """Test that digest generation is tracked."""
    from chinvex.digest import generate_digest
    from pathlib import Path

    # Get initial counter value
    before_count = _get_counter_value("chinvex_digest_generated_total", {"context": "TestContext"})

    # Generate digest
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        output_md = Path(tmpdir) / "digest.md"
        output_json = Path(tmpdir) / "digest.json"

        generate_digest(
            context_name="TestContext",
            ingest_runs_log=None,
            watch_history_log=None,
            state_md=None,
            output_md=output_md,
            output_json=output_json,
            since_hours=24
        )

    # Check metrics were incremented
    after_count = _get_counter_value("chinvex_digest_generated_total", {"context": "TestContext"})
    assert after_count == before_count + 1


def test_brief_generated_metric():
    """Test that brief generation is tracked."""
    from chinvex.brief import generate_brief
    from pathlib import Path

    # Get initial counter value
    before_count = _get_counter_value("chinvex_brief_generated_total", {"context": "TestContext"})

    # Generate brief
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "brief.md"

        generate_brief(
            context_name="TestContext",
            state_md=None,
            constraints_md=None,
            decisions_md=None,
            latest_digest=None,
            watch_history_log=None,
            output=output
        )

    # Check metrics were incremented
    after_count = _get_counter_value("chinvex_brief_generated_total", {"context": "TestContext"})
    assert after_count == before_count + 1


def _get_counter_value(metric_name: str, labels: dict) -> float:
    """Get current value of a counter metric with specific labels."""
    # Strip _total suffix if present, as prometheus_client adds it automatically
    base_name = metric_name.replace("_total", "")

    for metric in REGISTRY.collect():
        if metric.name == base_name:
            for sample in metric.samples:
                # Look for the _total sample (not _created)
                if sample.name == f"{base_name}_total":
                    # Check if labels match
                    if all(sample.labels.get(k) == v for k, v in labels.items()):
                        return sample.value
    return 0.0
