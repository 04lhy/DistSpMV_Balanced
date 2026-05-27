# ============================================================================
# run_exp.ps1 — Experiment launcher for DistSpMV_Balanced (C++) on Windows
# ============================================================================
#
# Usage:
#   .\scripts\run_exp.ps1 -Suite representative -Procs 1,2,4,8 -Threads 4
#   .\scripts\run_exp.ps1 -Matrix data/audikw_1.mtx -Procs 4 -Threads 4
#
# Prerequisites:
#   - MS-MPI installed (msmpisetup.exe + msmpisdk.msi)
#   - C++ build: mkdir build && cd build && cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release && mingw32-make -j
# ============================================================================

param(
    [string]$Procs = "1,2,4,8",
    [int]$Threads = 4,
    [string]$Matrix = "",
    [string]$Suite = "",
    [string]$Reorder = "rcm",
    [int]$Benchmark = 50,
    [int]$Warmup = 5,
    [int]$Seed = 42,
    [string]$OutDir = "results",
    [string]$Binary = ".\build\dist_spmv.exe",
    [string]$Mpiexec = "mpiexec"
)

$ErrorActionPreference = "Stop"

if ($args -contains "--help" -or $args -contains "-h") {
    Write-Host @"
Usage: run_exp.ps1 [OPTIONS]

Options:
  -Procs PROCS       Comma-separated list of process counts (default: 1,2,4,8)
  -Threads N         OMP_NUM_THREADS per rank (default: 4)
  -Matrix PATH       Single .mtx file to benchmark
  -Suite NAME        Predefined matrix suite: representative | paper
  -Reorder METHOD    Reordering: rcm | metis | none (default: rcm)
  -Benchmark N       Number of timed SpMV repetitions (default: 50)
  -Warmup N          Number of warmup calls (default: 5)
  -Seed N            Random seed (default: 42)
  -OutDir DIR        Output directory for JSON results (default: results)
  -Binary PATH       Path to dist_spmv binary
  -Mpiexec PATH      Path to mpiexec (default: mpiexec)
"@
    exit 0
}

# ── Matrix suites ─────────────────────────────────────────────────────
$SUITES = @{
    # Only matrices present in data/ are listed.  Download additional .mtx
    # files from SuiteSparse to expand this list (see README).
    "representative" = @("audikw_1")
    "paper" = @("cant", "road_central", "inline_1", "bone010", "pdb1HYS", "consph",
                 "cop20k_A", "torso1", "thermomech_dM", "hood", "bmwcra_1",
                 "torso2", "torso3", "Dubcova2", "qa8fm", "m_t1", "nd6k",
                 "rail_5177", "pwtk", "shipsec1")
}

if ($Suite) {
    if ($Suite -eq "all") { $Matrices = $SUITES["paper"] }
    else { $Matrices = $SUITES[$Suite] }
} elseif ($Matrix) {
    $Matrices = @([System.IO.Path]::GetFileNameWithoutExtension($Matrix))
    $DataDir = Split-Path $Matrix -Parent
} else {
    Write-Error "Specify -Matrix or -Suite"
    exit 1
}

# ── Verify binary ─────────────────────────────────────────────────────
if (-not (Test-Path $Binary)) {
    Write-Error "Binary not found: $Binary"
    Write-Host "  Build it first:"
    Write-Host "    mkdir build; cd build; cmake .. -G 'MinGW Makefiles' -DCMAKE_BUILD_TYPE=Release; mingw32-make -j"
    exit 1
}

# ── Environment ───────────────────────────────────────────────────────
$env:OMP_NUM_THREADS = $Threads
$env:OMP_PROC_BIND = "spread"
$env:OMP_PLACES = "threads"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "============================================"
Write-Host " DistSpMV_Balanced (C++) Experiment Runner"
Write-Host "============================================"
Write-Host " Binary    : $Binary"
Write-Host " MPI procs : $Procs"
Write-Host " Threads   : $Threads"
Write-Host " Reorder   : $Reorder"
Write-Host " Benchmark : $Benchmark iterations"
Write-Host " Seed      : $Seed"
Write-Host " Output    : $OutDir/"
Write-Host "============================================"

$ProcArray = $Procs -split '[, ]+' | Where-Object { $_ }

foreach ($mtx in $Matrices) {
    if (Test-Path "data/${mtx}.mtx") {
        $MtxPath = "data/${mtx}.mtx"
    } elseif ($DataDir -and (Test-Path "${DataDir}/${mtx}.mtx")) {
        $MtxPath = "${DataDir}/${mtx}.mtx"
    } else {
        Write-Warning "Matrix file not found for '$mtx' - skipping."
        Write-Host "  Download from https://sparse.tamu.edu/ and place in data/"
        continue
    }

    foreach ($np in $ProcArray) {
        Write-Host ""
        Write-Host ">>> Running: mtx=$mtx  np=$np  threads=$Threads"

        $OutFile = "${OutDir}/${mtx}_p${np}_t${Threads}.json"
        $LogFile = "${OutDir}/${mtx}_p${np}_t${Threads}.log"

        try {
            $cmd = "$Mpiexec -n $np `"$Binary`" --matrix `"$MtxPath`" --threads $Threads --reorder $Reorder --benchmark $Benchmark --warmup $Warmup --seed $Seed --output `"$OutFile`" 2>&1"
            cmd /c $cmd | Tee-Object -FilePath $LogFile

            if ($LASTEXITCODE -eq 0) {
                Write-Host "[OK]   Results -> $OutFile"
            } else {
                Write-Host "[FAIL] Exit code $LASTEXITCODE"
                Get-Content $LogFile | Select-Object -Last 20
            }
        } catch {
            Write-Host "[FAIL] $_"
            Get-Content $LogFile -ErrorAction SilentlyContinue | Select-Object -Last 20
        }
    }
}

Write-Host ""
Write-Host "All experiments complete.  Results in $OutDir/"
Write-Host "To generate plots: python scripts/plot_results.py --results $OutDir/"
