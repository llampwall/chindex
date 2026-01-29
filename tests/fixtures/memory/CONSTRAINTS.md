# Constraints

## Infrastructure
- ChromaDB batch limit: 5000 vectors
- Embedding dims locked per index (see meta.json)
- Gateway port: 7778
- Contexts root: P:\ai_memory\contexts

## Rules
- Schema stays v2 - no migrations without rebuild
- Metrics endpoint requires auth
- Archive is dry-run by default
- Index metadata (meta.json) is source of truth for dimensions

## Key Facts
- Gateway: localhost:7778 → chinvex.unkndlabs.com
- Token env var: CHINVEX_API_TOKEN
- OpenAI API key: OPENAI_API_KEY
