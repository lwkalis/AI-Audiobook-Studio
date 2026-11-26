Attempting to perform the InitializeDefaultDrives operation on the 'FileSystem' provider failed.
param(
    [ValidateSet('chatterbox', 'qwen')]
    [string[]]$Engine = @('chatterbox', 'qwen')
)

$ErrorActionPreference = 'Stop'
$runtime = Join-Path $PSScriptRoot '.runtime'
$env:HF_HOME = Join-Path $runtime 'hf'
$env:HF_HUB_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'

foreach ($name in $Engine) {
    $env:NUMBA_CACHE_DIR = Join-Path $runtime "numba-$name"
    $python = Join-Path $runtime "$name\Scripts\python.exe"
    & $python -u (Join-Path $PSScriptRoot 'benchmark.py') $name
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

