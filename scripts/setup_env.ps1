param(
    [ValidateSet("minimal", "full")]
    [string]$Profile = "full",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

# 无论从哪个目录调用脚本，都在仓库根目录创建 .venv。
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvDir = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "正在创建 Python 虚拟环境：$venvDir"
    & $Python -m venv $venvDir
}

Write-Host "正在更新 pip、setuptools 和 wheel"
& $venvPython -m pip install --upgrade pip setuptools wheel

if ($Profile -eq "minimal") {
    # minimal 只安装特征参考实现、测试和代码检查工具。
    $installTarget = ".[dev]"
} else {
    # full 覆盖训练、评估、绘图、TFLite 转换、测试和代码检查。
    $installTarget = ".[all]"
}

Write-Host "正在安装 $Profile 环境依赖"
Push-Location $projectRoot
try {
    & $venvPython -m pip install --editable $installTarget
    & $venvPython -m pytest
} finally {
    Pop-Location
}

Write-Host "虚拟环境配置完成。激活命令：.\.venv\Scripts\Activate.ps1"

