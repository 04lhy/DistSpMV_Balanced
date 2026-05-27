#pragma once
/**
 * Utility functions: row partitioning, error computation, GFlops, logging.
 */

#include <cstdio>
#include <string>
#include <utility>

#include "types.hpp"

namespace distspmv {

// ── Row distribution ─────────────────────────────────────────────────

/// Return (start, end) for contiguous 1-D row distribution across nprocs.
inline std::pair<idx_t, idx_t> row_partition_range(idx_t nrows, int nprocs,
                                                    int rank) {
    idx_t base = nrows / static_cast<idx_t>(nprocs);
    idx_t rem = nrows % static_cast<idx_t>(nprocs);
    if (rank < static_cast<int>(rem)) {
        idx_t start = rank * (base + 1);
        return {start, start + base + 1};
    } else {
        idx_t start = static_cast<idx_t>(rank) * base + rem;
        return {start, start + base};
    }
}

// ── Error / verification ─────────────────────────────────────────────

/// Relative L2 error: ||y_test - y_ref||_2 / ||y_ref||_2.
double compute_error(const double* y_test, const double* y_ref, idx_t n);

/// Serial SpMV: y = A * x.  Used for verification.
void serial_spmv(const CSRMatrix& A, const double* x, double* y);

// ── GFlops ───────────────────────────────────────────────────────────

/// Compute GFlops = (2 * nnz) / (time_sec * 1e9).
inline double gflops(int64_t nnz, double time_sec) {
    if (time_sec <= 0.0) return 0.0;
    return (2.0 * static_cast<double>(nnz)) / (time_sec * 1e9);
}

// ── Logging ──────────────────────────────────────────────────────────

/// Print a timestamped log message with rank prefix.
void log_info(int rank, const char* fmt, ...);

/// Set random seed for reproducibility.
void set_seed(int seed);

}  // namespace distspmv
