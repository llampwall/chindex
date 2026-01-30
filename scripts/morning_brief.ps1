<#
.SYNOPSIS
    Generate and send morning brief with context status

.DESCRIPTION
    Aggregates STATUS.json from all contexts and sends ntfy push.
    Writes MORNING_BRIEF.md to configured output path.

.PARAMETER ContextsRoot
    Path to contexts root directory

.PARAMETER NtfyTopic
    ntfy.sh topic for morning brief

.PARAMETER NtfyServer
    ntfy server URL (default: https://ntfy.sh)

.PARAMETER OutputPath
    Path to write MORNING_BRIEF.md (default: contexts root parent)

.EXAMPLE
    .\morning_brief.ps1 -ContextsRoot "P:\ai_memory\contexts" -NtfyTopic "morning-brief"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ContextsRoot,

    [Parameter(Mandatory=$false)]
    [string]$NtfyTopic = "",

    [Parameter(Mandatory=$false)]
    [string]$NtfyServer = "https://ntfy.sh",

    [Parameter(Mandatory=$false)]
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Continue"

# Default output path
if (-not $OutputPath) {
    $OutputPath = Join-Path (Split-Path $ContextsRoot -Parent) "MORNING_BRIEF.md"
}

# Collect all STATUS.json files
$contexts = Get-ChildItem -Path $ContextsRoot -Directory
$statusData = @()

foreach ($ctx in $contexts) {
    $statusFile = Join-Path $ctx.FullName "STATUS.json"
    if (Test-Path $statusFile) {
        try {
            $status = Get-Content $statusFile | ConvertFrom-Json
            $statusData += $status
        } catch {
            Write-Warning "Failed to parse ${statusFile}: $_"
        }
    }
}

# Calculate totals
$totalChunks = ($statusData | Measure-Object -Property chunks -Sum).Sum
$staleContexts = $statusData | Where-Object { $_.freshness.is_stale -eq $true }
$pendingWatches = ($statusData | Measure-Object -Property watches_pending_hits -Sum).Sum

# Generate markdown
$timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
$markdown = @"
# Dual Nature Morning Brief
Generated: $timestamp

## Summary
- **Total Contexts:** $($statusData.Count)
- **Total Chunks:** $($totalChunks.ToString("N0"))
- **Stale Contexts:** $($staleContexts.Count)
- **Pending Watch Hits:** $pendingWatches

## Context Details
| Context | Chunks | Last Sync | Status |
|---------|--------|-----------|--------|
"@

foreach ($status in ($statusData | Sort-Object -Property context)) {
    $statusIcon = if ($status.freshness.is_stale) { "[STALE]" } else { "[OK]" }
    $lastSync = if ($status.last_sync) {
        $syncTime = [datetime]::Parse($status.last_sync)
        $hoursAgo = [math]::Round(((Get-Date) - $syncTime).TotalHours, 1)
        "${hoursAgo}h ago"
    } else {
        "unknown"
    }

    $markdown += "`n| $($status.context) | $($status.chunks.ToString("N0")) | $lastSync | $statusIcon |"
}

if ($staleContexts.Count -gt 0) {
    $markdown += "`n`n## Stale Contexts"
    foreach ($ctx in $staleContexts) {
        $markdown += "`n- **$($ctx.context)**: $($ctx.freshness.hours_since_sync) hours since sync"
    }
}

if ($pendingWatches -gt 0) {
    $markdown += "`n`n## Pending Watch Hits"
    $markdown += "`nTotal pending: $pendingWatches"
}

# Write markdown file
$markdown | Out-File -FilePath $OutputPath -Encoding UTF8
Write-Host "Wrote brief to $OutputPath"

# Send ntfy push
if ($NtfyTopic) {
    $title = "Dual Nature Morning Brief"
    $body = "Contexts: $($statusData.Count) ($($staleContexts.Count) stale)`nChunks: $($totalChunks.ToString("N0"))`nWatch hits: $pendingWatches"

    if ($staleContexts.Count -gt 0) {
        $staleNames = ($staleContexts | ForEach-Object { $_.context }) -join ", "
        $body += "`nStale: $staleNames"
    }

    try {
        $url = "$NtfyServer/$NtfyTopic"
        $headers = @{
            "Title" = $title
            "Tags" = "sunrise,calendar"
        }

        Invoke-RestMethod -Uri $url -Method Post -Body $body -Headers $headers | Out-Null
        Write-Host "Sent morning brief push"
    } catch {
        Write-Warning "Failed to send ntfy push: $_"
    }
}
