# ============================================================================
# run_exp.ps1 — DistSpMV_Balanced experiment runner (PowerShell version)
# ============================================================================
#
# Usage:
#   .\scripts\run_exp.ps1 -Suite representative -Procs 1,2,4,8 -Threads 4
#
# ============================================================================

param(
    [string]$Procs = "1,2,4,8",
    [int]$Threads = 4,
    [string]$Matrix = "",
    [string]$Suite = "",
    [string]$Reorder = "rcm",
    [int]$Benchmark = 10,
    [int]$Warmup = 5,
    [int]$Seed = 42,
    [string]$OutDir = "results"
)

$env:OMP_NUM_THREADS = $Threads
$env:OPENBLAS_NUM_THREADS = 1
$env:MKL_NUM_THREADS = 1
$env:NUMEXPR_NUM_THREADS = 1

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# ── Matrix suite definitions ──
$Suites = @{
    "representative" = @("audikw_1")
    "paper" = @("cant", "road_central", "inline_1", "bone010", "pdb1HYS", "consph",
                 "cop20k_A", "torso1", "thermomech_dM", "hood", "bmwcra_1",
                 "torso2", "torso3", "Dubcova2", "qa8fm", "m_t1", "nd6k",
                 "rail_5177", "pwtk", "shipsec1")
}

# ── Resolve matrix list ──
if ($Suite) {
    if ($Suite -eq "all") { $Matrices = $Suites["paper"] }
    else { $Matrices = $Suites[$Suite] }
} elseif ($Matrix) {
    $Matrices = @([System.IO.Path]::GetFileNameWithoutExtension($Matrix))
    $DataDir = Split-Path $Matrix -Parent
} else {
    Write-Error "Specify -Matrix or -Suite"
    exit 1
}

Write-Host "============================================"
Write-Host " DistSpMV_Balanced Experiment Runner"
Write-Host "============================================"
Write-Host " MPI procs : $Procs"
Write-Host " Threads   : $Threads"
Write-Host " Reorder   : $Reorder"
Write-Host " Benchmark : $Benchmark iterations"
Write-Host " Seed      : $Seed"
Write-Host " Output    : $OutDir/"
Write-Host "============================================"

$ProcArray = $Procs -split "," | ForEach-Object { $_.Trim() }

foreach ($mtx in $Matrices) {
    # Find matrix file
    $mtxPath = "data/$mtx.mtx"
    if (-not (Test-Path $mtxPath)) {
        Write-Warning "Matrix file not found: $mtxPath — skipping"
        continue
    }

    foreach ($np in $ProcArray) {
        Write-Host ""
        Write-Host ">>> Running: mtx=$mtx  np=$np  threads=$Threads"

        $outFile = "$OutDir/${mtx}_p${np}_t${Threads}.json"
        $logFile = "$OutDir/${mtx}_p${np}_t${Threads}.log"

        try {
            # Wrap stderr→stdout via cmd /c to avoid NativeCommandError (red text)
            $cmd = "mpiexec -n $np python -m src.main --matrix $mtxPath --threads $Threads --reorder $Reorder --benchmark $Benchmark --warmup $Warmup --seed $Seed --output $outFile 2>&1"
            cmd /c $cmd | Tee-Object -FilePath $logFile

            if ($LASTEXITCODE -eq 0) {
                Write-Host "[OK]   Results -> $outFile"
            } else {
                Write-Host "[FAIL] Exit code $LASTEXITCODE"
                Get-Content $logFile | Select-Object -Last 20
            }
        } catch {
            Write-Host "[FAIL] $_"
            Get-Content $logFile -ErrorAction SilentlyContinue | Select-Object -Last 20
        }
    }
}

Write-Host ""
Write-Host "All experiments complete.  Results in $OutDir/"
Write-Host "To generate plots: python scripts/plot_results.py --results $OutDir/"
