Attempting to perform the InitializeDefaultDrives operation on the 'FileSystem' provider failed.
$ErrorActionPreference = 'Stop'
$runtime = Join-Path $PSScriptRoot '.runtime'
$python = Join-Path $runtime 'qwen\Scripts\python.exe'
$model = Join-Path $runtime 'models\qwen\model.safetensors'
if (!(Test-Path -LiteralPath $python) -or !(Test-Path -LiteralPath $model)) {
    throw 'The local Qwen runtime or model is missing. See README.md.'
}
$env:NUMBA_CACHE_DIR = Join-Path $runtime 'numba-qwen'
& $python -u (Join-Path $PSScriptRoot 'app.py') @args
exit $LASTEXITCODE

