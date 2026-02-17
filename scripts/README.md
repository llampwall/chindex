# Scripts

PowerShell and Python scripts for chinvex maintenance and operations.

## Memory Maintenance

### `ingest-memory-files.ps1`
Bulk-ingests `docs/memory/*.md` files for all managed repos into their respective chinvex contexts.

Use this when you've made bulk edits to memory files across multiple repos and want to re-index them immediately without waiting for the sync daemon.

```powershell
.\scripts\ingest-memory-files.ps1
```

### `touch-memory-files.ps1`
Sets `LastWriteTime = now` on `docs/memory/*.md` files across all managed repos.

Use this to force the file watcher daemon to pick up memory files on its next sweep, without running a full ingest manually.

```powershell
.\scripts\touch-memory-files.ps1
```

> **Workflow:** Use `touch-memory-files.ps1` to trigger re-ingest via the daemon, or `ingest-memory-files.ps1` to do it immediately.

---

## Gateway & Services

### `start_mcp.ps1`
Starts the chinvex MCP server. Reads the API token from `~/.secrets/chinvex_token`.

```powershell
.\scripts\start_mcp.ps1
```

### `start_pm2_services.ps1`
PM2 startup script registered as a Windows scheduled task. Runs at logon, waits 5 seconds for network, then calls `pm2 resurrect` to bring up all PM2 services (gateway + cloudflared tunnel).

Normally runs automatically — invoke manually to recover after an unexpected shutdown:
```powershell
.\scripts\start_pm2_services.ps1
```

### `update_scheduled_task.ps1`
Registers (or re-registers) the `PM2 Services Startup` Windows scheduled task pointing to `start_pm2_services.ps1`. Run this once after changing the startup script or moving the repo.

```powershell
.\scripts\update_scheduled_task.ps1
# Test with:
Start-ScheduledTask -TaskName "PM2 Services Startup"
```

---

## Scheduled Automation

### `scheduled_sweep.ps1`
Runs every 30 minutes via Task Scheduler. Ensures the watcher daemon is running (detects zombie processes via heartbeat), then runs an ingest sweep across all contexts.

Parameters:
- `-ContextsRoot` (required) — path to contexts root
- `-NtfyTopic` (optional) — ntfy.sh topic for alerts
- `-NtfyServer` (optional) — ntfy server URL (default: `https://ntfy.sh`)

### `scheduled_sweep_launcher.vbs`
VBScript wrapper that launches `scheduled_sweep.ps1` without a visible PowerShell window. Used as the Task Scheduler action to avoid a flash of terminal on each run.

### `morning_brief.ps1`
Generates a morning brief across all contexts and optionally sends it as an ntfy push notification.

Parameters:
- `-ContextsRoot` (required) — path to contexts root
- `-NtfyTopic` (optional) — ntfy.sh topic for delivery
- `-NtfyServer` (optional) — ntfy server URL (default: `https://ntfy.sh`)
- `-OutputPath` (optional) — path to write `MORNING_BRIEF.md`

```powershell
.\scripts\morning_brief.ps1 -ContextsRoot "P:\ai_memory\contexts" -NtfyTopic "my-topic"
```

---

## Backup

### `backup.ps1`
Snapshots `P:\ai_memory\contexts` and `P:\ai_memory\indexes` to `P:\backups\chinvex\<date>`. Run before major migrations or index rebuilds.

```powershell
.\scripts\backup.ps1
```

---

## E2E Smoke Tests

Python scripts for end-to-end smoke testing across implementation phases.

| Script | Phase | Coverage |
|--------|-------|----------|
| `e2e_smoke_p0.py` | P0 | Core ingest and FTS search |
| `e2e_smoke_p1.py` | P1 | Vector search and hybrid ranking |
| `e2e_smoke_p2.py` | P2 | Gateway API endpoints |
| `e2e_smoke_p3.py` | P3 | Sync daemon and file watching |
| `e2e_smoke_p4.py` | P4 | Context management and purge |

Run with:
```powershell
python .\scripts\e2e_smoke_p0.py
```

### `test_gateway_p2.py`
Targeted smoke test for the gateway (P2). Useful for quick gateway health validation without running the full suite.

### `test_scheduled_task_path.ps1`
Verifies that the scheduled task environment has the correct PATH (includes pnpm, Python). Run if PM2 resurrect fails silently after system restart.
