# PM2 Scheduled Task Setup

## Problem
The scheduled task running `pm2 resurrect` fails because it can't find `pm2` in the PATH.

## Solution
Use a PowerShell script that sets up the environment and calls pm2 with the full path.

## Setup Steps

Run these commands in **PowerShell as Administrator**:

```powershell
# 1. Remove old scheduled task
Unregister-ScheduledTask -TaskName "PM2 Resurrect" -Confirm:$false -ErrorAction SilentlyContinue

# 2. Create new scheduled task
$action = New-ScheduledTaskAction -Execute "pwsh.exe" -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"P:\software\chinvex\scripts\start_pm2_services.ps1`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "PM2 Services Startup" -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest -Description "Starts PM2 services at logon"

# 3. Test the task
pm2 kill
Start-Sleep -Seconds 2
Start-ScheduledTask -TaskName "PM2 Services Startup"
Start-Sleep -Seconds 10
pm2 list
```

## Verify

After running the setup, check:
1. Services are running: `pm2 list`
2. Startup log was created: `Get-Content C:\Users\Jordan\.pm2\startup.log`
3. Scheduled task exists: `Get-ScheduledTask -TaskName "PM2 Services Startup"`

## Manual Test

To manually test the startup script:
```powershell
pm2 kill
pwsh -ExecutionPolicy Bypass -File P:\software\chinvex\scripts\start_pm2_services.ps1
pm2 list
```

## What This Fixes

- **Old approach**: Scheduled task runs `pm2 resurrect` directly → pm2 not found in PATH → fails silently
- **New approach**: Scheduled task runs PowerShell script → script sets up PATH and calls pm2 with full path → works reliably

## Files Created

- `P:\software\chinvex\scripts\start_pm2_services.ps1` - Startup script
- `C:\Users\Jordan\.pm2\startup.log` - Logs startup attempts (useful for debugging)
