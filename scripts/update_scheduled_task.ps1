# Update PM2 Scheduled Task
# This script removes the old task and creates a new one using the startup script

# Remove old scheduled task
Unregister-ScheduledTask -TaskName "PM2 Resurrect" -Confirm:$false -ErrorAction SilentlyContinue

# Create new scheduled task
$action = New-ScheduledTaskAction `
    -Execute "pwsh.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"P:\software\chinvex\scripts\start_pm2_services.ps1`""

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask `
    -TaskName "PM2 Services Startup" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -RunLevel Highest `
    -Description "Starts PM2 services (chinvex-gateway and chinvex-tunnel) at logon"

Write-Host "Scheduled task updated successfully!"
Write-Host "Task name: PM2 Services Startup"
Write-Host "Test the task with: Start-ScheduledTask -TaskName 'PM2 Services Startup'"
