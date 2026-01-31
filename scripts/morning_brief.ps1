<#
.SYNOPSIS
    Generate and send morning brief with context status

.DESCRIPTION
    Uses chinvex.morning_brief Python module to generate brief with
    active project objectives and send ntfy push.

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

$ErrorActionPreference = "Stop"

# Default output path
if (-not $OutputPath) {
    $OutputPath = Join-Path (Split-Path $ContextsRoot -Parent) "MORNING_BRIEF.md"
}

# Call Python module to generate brief
$pythonCode = @"
import sys
from pathlib import Path
from chinvex.morning_brief import generate_morning_brief

contexts_root = Path(r'$ContextsRoot')
output_path = Path(r'$OutputPath')

brief_text, ntfy_body = generate_morning_brief(contexts_root, output_path)

# Print ntfy body to stdout for PowerShell to capture
print(ntfy_body, end='')
"@

try {
    # Run Python code and capture ntfy body
    $ntfyBody = python -c $pythonCode

    Write-Host "Generated morning brief at $OutputPath"

    # Send ntfy push if topic is configured
    if ($NtfyTopic) {
        $title = "Morning Brief"

        $url = "$NtfyServer/$NtfyTopic"
        $headers = @{
            "Title" = $title
            "Tags" = "sunrise,calendar"
        }

        Invoke-RestMethod -Uri $url -Method Post -Body $ntfyBody -Headers $headers | Out-Null
        Write-Host "Sent morning brief push to $NtfyTopic"
    }
} catch {
    Write-Error "Failed to generate morning brief: $_"
    exit 1
}
