$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Frontend = Join-Path $Root "frontend"
$ApiBase = if ($env:VITE_API_BASE) { $env:VITE_API_BASE } else { "http://127.0.0.1:8010" }
$env:VITE_API_BASE = $ApiBase

if (-not (Test-Path (Join-Path $Frontend "node_modules"))) {
  Write-Host "Installing frontend dependencies..."
  Push-Location $Frontend
  npm install
  Pop-Location
}

Write-Host "Using VITE_API_BASE=$ApiBase"
Write-Host "Starting EnergyPlus-Agent API at http://127.0.0.1:8010"
$Api = Start-Process -FilePath "python" `
  -ArgumentList "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "8010" `
  -WorkingDirectory $Root `
  -PassThru `
  -WindowStyle Hidden

Start-Sleep -Seconds 2

try {
  $Health = Invoke-RestMethod -Uri "$ApiBase/api/models" -TimeoutSec 10
  $ModelCount = @($Health.models).Count
  Write-Host "API ready: /api/models returned $ModelCount model(s)."
}
catch {
  Write-Warning "API health check failed: $($_.Exception.Message)"
}

Write-Host "Starting frontend at http://127.0.0.1:5173"
Push-Location $Frontend
try {
  npm run dev
}
finally {
  Pop-Location
  if ($Api -and -not $Api.HasExited) {
    Stop-Process -Id $Api.Id -Force
  }
}
