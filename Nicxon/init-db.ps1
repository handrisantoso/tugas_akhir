# ===============================================================
#  init-db.ps1 — One-time vector database volume initialiser
#
#  Run this ONCE before your first "docker compose up".
#  It copies each ChromaDB folder from your local machine into
#  the named Docker volumes that the container will use.
#
#  Usage:
#    .\init-db.ps1
# ===============================================================

$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

$dbs = @(
    @{ name = "compressed_chroma_google"; src = "vector_db\chroma_db_multimodal_google" },
    @{ name = "compressed_chroma_jina";   src = "vector_db\chroma_db_multimodal_jina"   },
    @{ name = "compressed_chroma_qwen";   src = "vector_db\chroma_db_multimodal_qwen"   }
)

Write-Host ""
Write-Host "=== Library RAG — Vector DB Volume Initialiser ===" -ForegroundColor Cyan
Write-Host ""

foreach ($db in $dbs) {
    $srcPath = Join-Path $ROOT $db.src
    $volName = $db.name

    if (-not (Test-Path $srcPath)) {
        Write-Host "SKIP  $volName — source folder not found: $srcPath" -ForegroundColor Yellow
        continue
    }

    Write-Host "Copying $($db.src)  ->  volume: $volName ..." -ForegroundColor White

    # Use a temporary Alpine container to receive the copy.
    # The volume is mounted at /data; we pipe a tar stream into it.
    $result = docker run --rm `
        -v "${volName}:/data" `
        -v "${srcPath}:/src:ro" `
        alpine `
        sh -c "cp -r /src/. /data/ && echo OK"

    if ($result -eq "OK") {
        Write-Host "  Done." -ForegroundColor Green
    } else {
        Write-Error "  Failed to populate volume $volName. Is Docker running?"
    }
}

Write-Host ""
Write-Host "All volumes ready. Run: docker compose up" -ForegroundColor Cyan
Write-Host ""

