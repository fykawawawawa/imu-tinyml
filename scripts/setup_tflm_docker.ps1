[CmdletBinding()]
param(
    [switch]$SkipSmokeTest
)

$ErrorActionPreference = "Stop"
$TflmRef = "cd32a519356a05615e58001637ee25f73b009c86"

Write-Host "Checking Docker Linux engine..."
$serverOs = docker info --format "{{.OSType}}"
if ($LASTEXITCODE -ne 0 -or $serverOs.Trim() -ne "linux") {
    throw "Docker Linux engine is unavailable. Start Docker Desktop with the WSL 2 backend."
}

Write-Host "Pulling the pinned TFLM toolchain image..."
docker compose pull tflm
if ($LASTEXITCODE -ne 0) {
    throw "Failed to pull the TFLM toolchain image."
}

$smokeTest = if ($SkipSmokeTest) {
    "echo 'Skipping the TFLM hello_world smoke test.'"
} else {
    @"
make -f tensorflow/lite/micro/tools/make/Makefile \
    -j2 \
    test_hello_world_test
"@
}

$bootstrap = @"
set -euo pipefail

if [ ! -d /opt/tflm/.git ]; then
    if [ -n "`$(find /opt/tflm -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo '/opt/tflm is not empty and is not a Git repository.' >&2
        exit 1
    fi
    git clone https://github.com/tensorflow/tflite-micro.git /opt/tflm
fi

cd /opt/tflm
git fetch origin $TflmRef --depth 1
git checkout --detach $TflmRef
echo "TFLM commit: `$(git rev-parse HEAD)"
$smokeTest
"@

Write-Host "Preparing the pinned TFLM source tree..."
docker compose run --rm tflm bash -lc $bootstrap
if ($LASTEXITCODE -ne 0) {
    throw "TFLM container setup failed."
}

Write-Host "TFLM Docker environment is ready."
Write-Host "Open a shell with: docker compose run --rm tflm bash"
