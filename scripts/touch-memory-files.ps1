$repos = @('allmind', 'chinvex', 'codex_bot', 'FootballTracker', 'godex', 'mobile-comfy', 'rayban-voice-assistant', 'sentinel-kit', 'streamside', 'tapout-recovery-landing-page', 'unclaimed-asset-manager', 'VisoMaster', '_strap')

foreach ($repo in $repos) {
    $memPath = "P:\software\$repo\docs\memory"
    if (Test-Path $memPath) {
        Get-ChildItem "$memPath\*.md" | ForEach-Object {
            $_.LastWriteTime = Get-Date
            Write-Host "Touched: $($_.FullName)"
        }
    }
}
