# PM2 Services Startup Script
# This script ensures PM2 services start reliably at system boot

# Add pnpm and node to PATH
$env:PATH = "C:\Users\Jordan\AppData\Local\pnpm;$env:PATH"

# Set PM2_HOME explicitly
$env:PM2_HOME = "C:\Users\Jordan\.pm2"

# Log startup attempt
$logFile = "C:\Users\Jordan\.pm2\startup.log"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logFile -Value "[$timestamp] Starting PM2 services..."

# Wait a bit for network to be ready (important for cloudflared)
Start-Sleep -Seconds 5

# Resurrect PM2 processes
try {
    & "C:\Users\Jordan\AppData\Local\pnpm\pm2" resurrect
    Add-Content -Path $logFile -Value "[$timestamp] PM2 resurrect completed successfully"
} catch {
    Add-Content -Path $logFile -Value "[$timestamp] ERROR: $($_.Exception.Message)"
}
