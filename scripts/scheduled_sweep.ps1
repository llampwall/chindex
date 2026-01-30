<#
.SYNOPSIS
    Scheduled sweep - ensures watcher running and syncs all contexts

.DESCRIPTION
    Runs every 30 minutes via Task Scheduler.
    - Ensures watcher daemon is running
    - Checks watcher heartbeat (detects zombie processes)
    - Runs ingest sweep for all contexts
    - Archives _global context if needed

.PARAMETER ContextsRoot
    Path to contexts root directory

.PARAMETER NtfyTopic
    ntfy.sh topic for alerts (optional)

.PARAMETER NtfyServer
    ntfy server URL (default: https://ntfy.sh)

.PARAMETER StateDir
    State directory for watcher (default: ~/.chinvex)

.EXAMPLE
    .\scheduled_sweep.ps1 -ContextsRoot "P:\ai_memory\contexts" -NtfyTopic "chinvex-alerts"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ContextsRoot,

    [Parameter(Mandatory=$false)]
    [string]$NtfyTopic = "",

    [Parameter(Mandatory=$false)]
    [string]$NtfyServer = "https://ntfy.sh",

    [Parameter(Mandatory=$false)]
    [string]$StateDir = (Join-Path $env:USERPROFILE ".chinvex")
)

$ErrorActionPreference = "Continue"
$LogFile = Join-Path $env:USERPROFILE ".chinvex\sweep.log"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage
    Add-Content -Path $LogFile -Value $logMessage -ErrorAction SilentlyContinue
}

function Send-Alert {
    param([string]$Message)
    if ($NtfyTopic) {
        try {
            $url = "$NtfyServer/$NtfyTopic"
            Invoke-RestMethod -Uri $url -Method Post -Body $Message -ErrorAction Stop | Out-Null
            Write-Log "Alert sent: $Message"
        } catch {
            Write-Log "Failed to send alert: $_"
        }
    }
}

Write-Log "=== Sweep started ==="

# 1. Ensure watcher is running
Write-Log "Checking watcher status..."
try {
    $statusOutput = chinvex sync status --state-dir $StateDir 2>&1 | Out-String
    if ($statusOutput -match "NOT RUNNING") {
        Write-Log "Watcher not running, starting..."
        chinvex sync start --state-dir $StateDir
        Send-Alert "Chinvex watcher was down, restarted"
    } elseif ($statusOutput -match "STALE") {
        Write-Log "Watcher heartbeat stale, restarting..."
        chinvex sync stop --state-dir $StateDir
        Start-Sleep -Seconds 2
        chinvex sync start --state-dir $StateDir
        Send-Alert "Chinvex watcher heartbeat stale, restarted"
    } else {
        Write-Log "Watcher running normally"
    }
} catch {
    Write-Log "Error checking watcher: $_"
    Send-Alert "Chinvex sweep: watcher check failed"
}

# 2. Reconcile sources (ensure watcher watching correct paths)
Write-Log "Reconciling sources..."
try {
    chinvex sync reconcile-sources --contexts-root $ContextsRoot 2>&1 | Out-Null
} catch {
    Write-Log "Source reconciliation failed: $_"
}

# 3. Sweep all contexts
Write-Log "Running ingest sweep..."
try {
    $sweepOutput = chinvex ingest --all-contexts --changed-only --skip-locked --contexts-root $ContextsRoot 2>&1
    Write-Log "Sweep output: $sweepOutput"
} catch {
    Write-Log "Sweep failed: $_"
    Send-Alert "Chinvex sweep: ingest failed"
}

# 3.5. Check for stale contexts and send alerts
Write-Log "Checking for stale contexts..."
try {
    # Get list of context directories
    $contexts = Get-ChildItem -Path $ContextsRoot -Directory | Where-Object { $_.Name -notlike "_*" }

    foreach ($ctx in $contexts) {
        $statusFile = Join-Path $ctx.FullName "STATUS.json"
        if (Test-Path $statusFile) {
            try {
                $status = Get-Content $statusFile | ConvertFrom-Json
                if ($status.freshness.is_stale) {
                    # Use Python helper to check dedup and send if allowed
                    $logFile = Join-Path $env:USERPROFILE ".chinvex\push_log.jsonl"
                    python -c "from chinvex.notify import send_stale_alert; send_stale_alert('$($ctx.Name)', '$logFile', '$NtfyServer', '$NtfyTopic')" 2>&1 | Out-Null
                    Write-Log "Checked stale alert for $($ctx.Name)"
                }
            } catch {
                Write-Log "Error checking stale status for $($ctx.Name): $_"
            }
        }
    }
} catch {
    Write-Log "Stale context check failed: $_"
}

# 4. Generate global status
Write-Log "Generating global status..."
try {
    chinvex status --regenerate --contexts-root $ContextsRoot 2>&1 | Out-Null
    Write-Log "Global status regenerated"
} catch {
    Write-Log "Global status generation failed: $_"
}

# 5. Archive _global context if needed
Write-Log "Checking _global archive..."
try {
    $archiveOutput = chinvex archive --context _global --apply-constraints --quiet --contexts-root $ContextsRoot 2>&1
    if ($archiveOutput -match "archived") {
        Write-Log "Archived chunks in _global: $archiveOutput"
        Send-Alert "Chinvex: _global context archived old chunks"
    }
} catch {
    Write-Log "Archive check failed: $_"
}

Write-Log "=== Sweep complete ==="
