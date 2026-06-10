# Watcher: notify_training_complete.ps1
# Polls for python3.11 process; when none remain, writes a log and shows a popup.
# Usage: run this in background in the project environment.

$projectRoot = "C:\Users\mkart\major project"
$checkpoint = Join-Path $projectRoot "speech_forensics\checkpoints\audio_forensics.pt"
$logFile = Join-Path $projectRoot "training_complete_notification.txt"

Write-Output "[Watcher] Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $logFile -Append

while (Get-Process python3.11 -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 10
}

Start-Sleep -Seconds 2

$chkInfo = if (Test-Path $checkpoint) { (Get-Item $checkpoint).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss') } else { 'NOT FOUND' }
$msg = "TRAINING COMPLETE`nTime: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`nCheckpoint LastWrite: $chkInfo"

# Append to log
$msg | Out-File -FilePath $logFile -Append

# Try to show a GUI popup (may require interactive session)
try {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show($msg, 'Training Complete') | Out-Null
} catch {
    # Fallback: write to console
    Write-Output $msg | Out-File -FilePath $logFile -Append
}

# Also write a short success marker
"DONE" | Out-File -FilePath (Join-Path $projectRoot "training_complete_flag.txt") -Force
