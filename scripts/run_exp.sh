#!/usr/bin/env bash
# ============================================================================
# run_exp.sh — One-click experiment launcher for DistSpMV_Balanced (C++)
# ============================================================================
#
# Usage:
#   bash scripts/run_exp.sh --procs 8 --threads 4 --matrix data/cant.mtx
#   bash scripts/run_exp.sh --procs 1,2,4,8 --threads 4 --suite all
#
# Prerequisites:
#   - MPI implementation (OpenMPI / MPICH) with mpiexec on PATH
#   - C++ build: mkdir build && cd build && cmake .. && make
# ============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
PROCS="1,2,4,8"
THREADS=4
MATRIX=""
SUITE=""
REORDER="rcm"
BENCHMARK=50
WARMUP=5
SEED=42
OUTDIR="results"
BINARY="${BINARY:-./build/dist_spmv}"
MPIEXEC="${MPIEXEC:-mpiexec}"
MPI_ARGS="${MPI_ARGS:-}"

# ── Help ──────────────────────────────────────────────────────────────
usage() {
    cat << 'EOF'
Usage: run_exp.sh [OPTIONS]

Options:
  --procs PROCS       Comma-separated list of process counts (default: 1,2,4,8)
  --threads N         OMP_NUM_THREADS per rank (default: 4)
  --matrix PATH       Single .mtx file to benchmark
  --suite NAME        Predefined matrix suite: all | representative | paper
  --reorder METHOD    Reordering: rcm | metis | none (default: rcm)
  --benchmark N       Number of timed SpMV repetitions (default: 50)
  --warmup N          Number of warmup calls (default: 5)
  --seed N            Random seed (default: 42)
  --outdir DIR        Output directory for JSON results (default: results)
  --binary PATH       Path to dist_spmv binary (default: ./build/dist_spmv)
  --mpiexec PATH      Path to mpiexec (default: mpiexec)
  --help              Show this message

For both suites, download .mtx files into data/ first (see README).
"representative" currently = audikw_1; "paper" = 20 matrices.
EOF
    exit 0
}

# ── CLI parsing ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --procs)    PROCS="$2"; shift 2 ;;
        --threads)  THREADS="$2"; shift 2 ;;
        --matrix)   MATRIX="$2"; shift 2 ;;
        --suite)    SUITE="$2"; shift 2 ;;
        --reorder)  REORDER="$2"; shift 2 ;;
        --benchmark) BENCHMARK="$2"; shift 2 ;;
        --warmup)   WARMUP="$2"; shift 2 ;;
        --seed)     SEED="$2"; shift 2 ;;
        --outdir)   OUTDIR="$2"; shift 2 ;;
        --binary)   BINARY="$2"; shift 2 ;;
        --mpiexec)  MPIEXEC="$2"; shift 2 ;;
        --help)     usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── Matrix suite definitions ──────────────────────────────────────────
declare -A SUITE_MATRICES
SUITE_MATRICES["representative"]="audikw_1"
SUITE_MATRICES["paper"]="cant road_central inline_1 bone010 pdb1HYS consph cop20k_A torso1 thermomech_dM hood bmwcra_1 torso2 torso3 Dubcova2 qa8fm m_t1 nd6k rail_5177 pwtk shipsec1"

if [[ -n "$SUITE" ]]; then
    if [[ "$SUITE" == "all" ]]; then
        MATRICES="${SUITE_MATRICES["paper"]}"
    else
        MATRICES="${SUITE_MATRICES[$SUITE]:-}"
        if [[ -z "$MATRICES" ]]; then
            echo "ERROR: unknown suite '$SUITE'.  Choices: all, representative, paper"
            exit 1
        fi
    fi
elif [[ -n "$MATRIX" ]]; then
    MATRICES="$(basename "$MATRIX" .mtx)"
    DATADIR="$(dirname "$MATRIX")"
else
    echo "ERROR: specify --matrix or --suite"
    usage
fi

# ── Verify binary exists ──────────────────────────────────────────────
if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: Binary not found at $BINARY"
    echo "  Build it first:"
    echo "    mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────
export OMP_NUM_THREADS=$THREADS
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

mkdir -p "$OUTDIR"

echo "============================================"
echo " DistSpMV_Balanced (C++) Experiment Runner"
echo "============================================"
echo " Binary    : $BINARY"
echo " MPI procs : $PROCS"
echo " Threads   : $THREADS"
echo " Reorder   : $REORDER"
echo " Benchmark : $BENCHMARK iterations"
echo " Seed      : $SEED"
echo " Output    : $OUTDIR/"
echo "============================================"

# ── Locate the project root ───────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Run experiments ───────────────────────────────────────────────────
IFS=',' read -ra PROC_ARRAY <<< "$PROCS"

for mtx_name in $MATRICES; do
    if [[ -f "data/${mtx_name}.mtx" ]]; then
        MTX_PATH="data/${mtx_name}.mtx"
    elif [[ -f "${DATADIR:-}/${mtx_name}.mtx" ]]; then
        MTX_PATH="${DATADIR}/${mtx_name}.mtx"
    else
        echo "[WARN] Matrix file not found for '$mtx_name' — skipping."
        echo "  Download from https://sparse.tamu.edu/ and place in data/"
        continue
    fi

    for np in "${PROC_ARRAY[@]}"; do
        np=$(echo "$np" | xargs)
        echo ""
        echo ">>> Running: mtx=$mtx_name  np=$np  threads=$THREADS"

        OUT_FILE="${OUTDIR}/${mtx_name}_p${np}_t${THREADS}.json"
        LOG_FILE="${OUTDIR}/${mtx_name}_p${np}_t${THREADS}.log"

        $MPIEXEC -n "$np" $MPI_ARGS \
            "$BINARY" \
            --matrix "$MTX_PATH" \
            --threads "$THREADS" \
            --reorder "$REORDER" \
            --benchmark "$BENCHMARK" \
            --warmup "$WARMUP" \
            --seed "$SEED" \
            --output "$OUT_FILE" \
            2>&1 | tee "$LOG_FILE"

        RC=${PIPESTATUS[0]}
        if [[ $RC -ne 0 ]]; then
            echo "[FAIL] Exit code $RC — see $LOG_FILE"
        else
            echo "[OK]   Results → $OUT_FILE"
        fi
    done
done

echo ""
echo "All experiments complete.  Results in $OUTDIR/"
echo "To generate plots: python scripts/plot_results.py --results $OUTDIR/"
