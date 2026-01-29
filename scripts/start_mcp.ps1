# Start MCP server with token from secrets
$env:CHINVEX_API_TOKEN = Get-Content ~/.secrets/chinvex_token
chinvex-mcp
