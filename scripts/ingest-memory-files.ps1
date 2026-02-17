$repos = @(
    @{name='allmind'; path='P:\software\allmind'},
    @{name='chinvex'; path='P:\software\chinvex'},
    @{name='codex_bot'; path='P:\software\codex_bot'},
    @{name='FootballTracker'; path='P:\software\FootballTracker'},
    @{name='godex'; path='P:\software\godex'},
    @{name='mobile-comfy'; path='P:\software\mobile-comfy'},
    @{name='rayban-voice-assistant'; path='P:\software\rayban-voice-assistant'},
    @{name='sentinel-kit'; path='P:\software\sentinel-kit'},
    @{name='streamside'; path='P:\software\streamside'},
    @{name='tapout-recovery-landing-page'; path='P:\software\tapout-recovery-landing-page'},
    @{name='unclaimed-asset-manager'; path='P:\software\unclaimed-asset-manager'},
    @{name='VisoMaster'; path='P:\software\VisoMaster'},
    @{name='_strap'; path='P:\software\_strap'}
)

foreach ($repo in $repos) {
    $memPath = "$($repo.path)\docs\memory"
    if (Test-Path $memPath) {
        $files = Get-ChildItem "$memPath\*.md"
        if ($files) {
            $pathsArg = ($files | ForEach-Object { $_.FullName }) -join ','
            Write-Host "Ingesting memory files for $($repo.name)..."
            & chinvex ingest --context $repo.name --paths $pathsArg
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  ✓ Success: $($repo.name)" -ForegroundColor Green
            } else {
                Write-Host "  ✗ Failed: $($repo.name)" -ForegroundColor Red
            }
        }
    }
}

Write-Host "`nAll memory files ingested!"
