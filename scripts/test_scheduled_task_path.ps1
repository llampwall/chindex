# Test what PATH looks like in a scheduled task context
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$logFile = "C:\Users\Jordan\.pm2\scheduled_task_path_test.log"

Add-Content -Path $logFile -Value "=== Test run at $timestamp ==="
Add-Content -Path $logFile -Value "`nPATH in scheduled task context:"
Add-Content -Path $logFile -Value $env:PATH
Add-Content -Path $logFile -Value "`n--- Path entries (split) ---"
$env:PATH -split ';' | ForEach-Object { Add-Content -Path $logFile -Value $_ }
Add-Content -Path $logFile -Value "`n--- Checking for pm2 ---"
Add-Content -Path $logFile -Value "pm2 location: $(Get-Command pm2 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)"
Add-Content -Path $logFile -Value "`n--- Checking for pnpm in PATH ---"
Add-Content -Path $logFile -Value "pnpm path exists: $(Test-Path 'C:\Users\Jordan\AppData\Local\pnpm')"
Add-Content -Path $logFile -Value "pnpm in PATH: $($env:PATH -like '*pnpm*')"
