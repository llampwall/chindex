# Chinvex MCP Server

MCP server that gives Claude Desktop/Code access to Chinvex memory via HTTP API.

## Install

```bash
pip install -e .
```

Or with uv:
```bash
uv pip install -e .
```

## Configuration

Set environment variables:
- `CHINVEX_URL` - Gateway URL (default: `https://chinvex.yourdomain.com`)
- `CHINVEX_API_TOKEN` - Bearer token for authentication (required)

## Claude Desktop Setup

Add to `~/.config/claude/claude_desktop_config.json` (Linux) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "chinvex": {
      "command": "chinvex-mcp",
      "env": {
        "CHINVEX_URL": "https://chinvex.yourdomain.com",
        "CHINVEX_API_TOKEN": "your-token-here"
      }
    }
  }
}
```

Or if using Python directly:

```json
{
  "mcpServers": {
    "chinvex": {
      "command": "python",
      "args": ["-m", "chinvex_mcp.server"],
      "env": {
        "CHINVEX_URL": "https://chinvex.yourdomain.com",
        "CHINVEX_API_TOKEN": "your-token-here"
      }
    }
  }
}
```

## Tools

### chinvex_search
Search personal knowledge base with grounded retrieval.

```
query: "retry logic implementation"
contexts: "all"  # or specific context name
k: 8  # number of results
```

### chinvex_list_contexts
List available contexts to search.

### chinvex_get_chunks
Retrieve full chunk content by ID after searching.

```
context: "Chinvex"
chunk_ids: ["chunk_abc123", "chunk_def456"]
```

## Test

```bash
export CHINVEX_API_TOKEN="your-token"
python -c "from chinvex_mcp.server import get_client; print(get_client().get('/health').json())"
```
