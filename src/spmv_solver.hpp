#pragma once
/**
 * Algorithms 3 & 4: MPI Communication + Multi-threaded Local SpMV.
 *
 * Paper reference: Section III.D (Algorithm 3), Section III.E (Algorithm 4).
 *
 * The DistSpMVSolver holds precomputed state so that multiply(x_global)
 * can be called repeatedly without rebuilding communication structures.
 *
 * Unified x_buf layout:
 *   x_buf[0..diag_len)         — locally owned x elements
 *   x_buf[diag_len..diag_len+R) — remote x elements, packed by peer
 *
 * local_pos[j] maps each nonzero column to its position in x_buf,
 * eliminating branches in the SpMV inner loop.
 */

#include <mpi.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "types.hpp"

namespace distspmv {

class DistSpMVSolver {
public:
    /// Constructor: precomputes all lookup structures.
    DistSpMVSolver(const idx_t* rowptr, const idx_t* colidx, const val_t* val,
                   idx_t nlocal, idx_t r_start, idx_t r_end,
                   idx_t left, idx_t right,
                   const CommSchedule& sched,
                   MPI_Comm comm, int n_threads = 1);

    /// Execute one distributed SpMV: y_local = A_local @ x_global.
    /// x_global is the full global x vector (length = N).
    /// Returns the local portion of y = A @ x (length = nlocal).
    void multiply(const double* x_global, double* y_out);

    /// Benchmark: run multiply n_warmup times, then n_repeat times.
    /// Returns (avg_time_sec, std_time_sec).
    std::pair<double, double> multiply_benchmark(const double* x_global,
                                                  int n_warmup, int n_repeat);

    // ── Accessors ──
    idx_t nnz_diag() const;
    idx_t nnz_offdiag() const { return nnz_local_ - nnz_diag(); }

    idx_t nlocal() const { return nlocal_; }
    idx_t diag_len() const { return diag_len_; }
    idx_t total_remote() const { return total_remote_; }
    int64_t nnz_local() const { return nnz_local_; }
    const CommSchedule& schedule() const { return sched_; }

private:
    // ── Algorithm 3: MPI communication ──
    void exchange_remote(const double* x_global);

    // ── Algorithm 4: local SpMV ──
    void local_spmv(double* y_out);

    // ── CSR data ──
    std::vector<idx_t> rowptr_;
    std::vector<idx_t> colidx_;
    std::vector<val_t> val_;
    idx_t nlocal_;
    idx_t r_start_, r_end_;
    idx_t left_, right_;
    int64_t nnz_local_;
    idx_t diag_len_;

    // ── Communication schedule (from Algorithm 2) ──
    CommSchedule sched_;

    // ── Unified x_buf layout ──
    std::vector<double> x_buf_;
    idx_t total_remote_ = 0;
    std::unordered_map<int, idx_t> remote_offset_;
    std::vector<idx_t> local_pos_;

    // ── Partition info for column ownership lookups ──
    std::vector<idx_t> owner_r_start_;
    std::vector<idx_t> owner_r_end_;

    // ── MPI / OpenMP ──
    MPI_Comm comm_;
    int rank_, nprocs_;
    int n_threads_;

    // ── Thread partition ──
    std::vector<idx_t> thread_bounds_;

    // Helper: find owner of a global column
    int find_owner_fast(idx_t col) const;
};

}  // namespace distspmv
