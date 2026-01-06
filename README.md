# chinvex - chunked vector indexer

Hybrid retrieval index CLI (SQLite FTS5 + Chroma) powered by Ollama embeddings.

## Prereqs
- Python 3.12
- Ollama installed and running
- `ollama pull mxbai-embed-large`

## Install (venv required)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

## Config
Create a JSON config file:
```json
{
  "index_dir": "P:\\ai_memory\\indexes\\streamside",
  "ollama_host": "http://127.0.0.1:11434",
  "embedding_model": "mxbai-embed-large",
  "sources": [
    {"type": "repo", "name": "streamside", "path": "C:\\Code\\streamside"},
    {"type": "chat", "project": "Twitch", "path": "P:\\ai_memory\\projects\\Twitch\\chats"}
  ]
}
```

## Run
```powershell
chinvex ingest --config P:\ai_memory\profiles\streamside.json
chinvex search --config P:\ai_memory\profiles\streamside.json "your query"
```

## Troubleshooting
- FTS5 missing: install a Python build with SQLite FTS5 enabled.
- Ollama connection/model missing: ensure Ollama is running and `ollama pull mxbai-embed-large` completed.
- Windows path issues: use escaped backslashes in JSON or forward slashes.

## Smoke Test
```powershell
chinvex ingest --config path\to\config.json
chinvex search --config path\to\config.json "known token"
```
Expected: ingest creates `<index_dir>\hybrid.db` and `<index_dir>\chroma`, and search returns results.
