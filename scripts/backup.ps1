# Snapshot context registry + indexes + digests
$timestamp = Get-Date -Format "yyyy-MM-dd"
$dest = "P:\backups\chinvex\$timestamp"

New-Item -ItemType Directory -Force -Path $dest

Copy-Item -Recurse P:\ai_memory\contexts "$dest\contexts"
Copy-Item -Recurse P:\ai_memory\indexes "$dest\indexes"

Write-Host "Backup complete: $dest"
