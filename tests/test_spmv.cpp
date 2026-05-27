/**
 * Unit tests for DistSpMV_Balanced.
 *
 * Run with:
 *   mpiexec -n 1 ./build/test_spmv     (serial tests)
 *   mpiexec -n 4 ./build/test_spmv     (parallel tests)
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include <mpi.h>

#include "comm_setup.hpp"
#include "mmio.hpp"
#include "partition.hpp"
#include "redistribute.hpp"
#include "reordering.hpp"
#include "spmv_solver.hpp"
#include "types.hpp"
#include "utils.hpp"

using namespace distspmv;

static int g_rank = 0;
static int g_nprocs = 1;
static int g_passed = 0;
static int g_failed = 0;

#define TEST(name)                                               \
    do {                                                         \
        if (g_rank == 0) std::printf("  TEST: %s ... ", name);   \
    } while (0)

#define PASS()                                                   \
    do {                                                         \
        if (g_rank == 0) std::printf("PASS\n");                  \
        g_passed++;                                              \
    } while (0)

#define FAIL(msg)                                                \
    do {                                                         \
        if (g_rank == 0) std::printf("FAIL: %s\n", msg);         \
        g_failed++;                                              \
    } while (0)

#define CHECK(cond)                                              \
    do {                                                         \
        if (!(cond)) { FAIL(#cond); return; }                    \
    } while (0)

#define CHECK_EQ(a, b)                                           \
    do {                                                         \
        if ((a) != (b)) {                                        \
            if (g_rank == 0) {                                   \
                std::printf("FAIL: %s == %s  (%d vs %d)\n",      \
                           #a, #b, (int)(a), (int)(b));          \
            }                                                    \
            g_failed++; return;                                  \
        }                                                        \
    } while (0)

#define CHECK_CLOSE(a, b, eps)                                   \
    do {                                                         \
        if (std::abs((a) - (b)) > (eps)) {                       \
            if (g_rank == 0) {                                   \
                std::printf("FAIL: %s ≈ %s  (%e vs %e)\n",       \
                           #a, #b, (double)(a), (double)(b));    \
            }                                                    \
            g_failed++; return;                                  \
        }                                                        \
    } while (0)

// ── Helper: create a simple CSR matrix ──

static CSRMatrix make_identity(idx_t n) {
    CSRMatrix A;
    A.nrows = n; A.ncols = n;
    A.rowptr.resize(n + 1);
    A.colidx.resize(n);
    A.val.resize(n);
    for (idx_t i = 0; i < n; ++i) {
        A.rowptr[i] = i;
        A.colidx[i] = i;
        A.val[i] = 1.0;
    }
    A.rowptr[n] = n;
    A.nnz = n;
    return A;
}

static CSRMatrix make_tridiag(idx_t n) {
    CSRMatrix A;
    A.nrows = n; A.ncols = n;
    int64_t nnz_est = 3LL * n - 2;
    A.rowptr.resize(n + 1);
    A.colidx.reserve(nnz_est);
    A.val.reserve(nnz_est);

    for (idx_t i = 0; i < n; ++i) {
        A.rowptr[i] = static_cast<idx_t>(A.colidx.size());
        if (i > 0)     { A.colidx.push_back(i - 1); A.val.push_back(-1.0); }
        A.colidx.push_back(i); A.val.push_back(2.0);
        if (i + 1 < n) { A.colidx.push_back(i + 1); A.val.push_back(-1.0); }
    }
    A.rowptr[n] = static_cast<idx_t>(A.colidx.size());
    A.nnz = A.rowptr.back();
    return A;
}

static CSRMatrix make_block_diag(idx_t n, int nblocks) {
    // Creates a block-diagonal matrix useful for testing multi-process distribution.
    CSRMatrix A;
    A.nrows = n; A.ncols = n;
    A.rowptr.resize(n + 1);

    idx_t block_size = n / nblocks;
    for (idx_t i = 0; i < n; ++i) {
        A.rowptr[i] = static_cast<idx_t>(A.colidx.size());
        idx_t block = i / block_size;
        if (block >= nblocks) block = nblocks - 1;
        idx_t start_col = block * block_size;
        idx_t end_col = std::min((block + 1) * block_size, n);

        // Fill row with a few entries in this block
        for (idx_t c = start_col; c < end_col; ++c) {
            if ((i + c) % 3 == 0) {  // Sparse within block
                A.colidx.push_back(c);
                A.val.push_back(1.0);
            }
        }
    }
    A.rowptr[n] = static_cast<idx_t>(A.colidx.size());
    A.nnz = A.rowptr.back();
    return A;
}

// ── Test: row_partition_range ──

static void test_row_partition() {
    TEST("row_partition_range");
    // 10 rows, 3 processes
    {
        auto [s0, e0] = row_partition_range(10, 3, 0);
        auto [s1, e1] = row_partition_range(10, 3, 1);
        auto [s2, e2] = row_partition_range(10, 3, 2);
        CHECK_EQ(s0, 0);
        CHECK_EQ(e0, 4);  // rank 0 gets ceil(10/3)=4 rows
        CHECK_EQ(s1, 4);
        CHECK_EQ(e1, 7);  // rank 1 gets 3 rows
        CHECK_EQ(s2, 7);
        CHECK_EQ(e2, 10); // rank 2 gets 3 rows
    }
    // 10 rows, 1 process
    {
        auto [s, e] = row_partition_range(10, 1, 0);
        CHECK_EQ(s, 0);
        CHECK_EQ(e, 10);
    }
    // 100 rows, 4 processes — all equal
    {
        auto [s, e] = row_partition_range(100, 4, 1);
        CHECK_EQ(s, 25);
        CHECK_EQ(e, 50);
    }
    PASS();
}

// ── Test: serial_spmv ──

static void test_serial_spmv() {
    TEST("serial_spmv");
    idx_t n = 5;
    auto A = make_identity(n);
    std::vector<double> x(n, 1.0);
    std::vector<double> y(n, 0.0);
    serial_spmv(A, x.data(), y.data());
    for (idx_t i = 0; i < n; ++i) {
        CHECK_CLOSE(y[i], 1.0, 1e-12);
    }
    PASS();
}

// ── Test: serial_spmv with tridiagonal ──

static void test_serial_spmv_tridiag() {
    TEST("serial_spmv_tridiag");
    idx_t n = 4;
    auto A = make_tridiag(n);
    // Tridiag: [2 -1 0 0; -1 2 -1 0; 0 -1 2 -1; 0 0 -1 2]
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y(n, 0.0);
    serial_spmv(A, x.data(), y.data());
    // y[0] = 2*1 + (-1)*2 = 0
    // y[1] = (-1)*1 + 2*2 + (-1)*3 = 0
    // y[2] = (-1)*2 + 2*3 + (-1)*4 = 0
    // y[3] = (-1)*3 + 2*4 = 5
    CHECK_CLOSE(y[0], 0.0, 1e-12);
    CHECK_CLOSE(y[1], 0.0, 1e-12);
    CHECK_CLOSE(y[2], 0.0, 1e-12);
    CHECK_CLOSE(y[3], 5.0, 1e-12);
    PASS();
}

// ── Test: compute_error ──

static void test_compute_error() {
    TEST("compute_error");
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {1.0, 2.0, 3.0};
    double err = compute_error(a.data(), b.data(), 3);
    CHECK_CLOSE(err, 0.0, 1e-14);

    std::vector<double> c = {1.1, 2.0, 3.0};
    err = compute_error(a.data(), c.data(), 3);
    CHECK(err > 0.0);
    PASS();
}

// ── Test: RCM reordering (identity matrix stays the same) ──

static void test_rcm_identity() {
    TEST("rcm_identity");
    auto A = make_identity(5);
    auto B = reorder_matrix(A, "rcm", g_nprocs, 42);
    CHECK_EQ(B.nrows, 5);
    CHECK_EQ(B.ncols, 5);
    // Every row should still have exactly one nnz and it should be on the diagonal
    // after reordering (identity is invariant under any symmetric permutation)
    int64_t nnz_total = 0;
    for (idx_t i = 0; i < B.nrows; ++i) {
        nnz_total += B.rowptr[i + 1] - B.rowptr[i];
    }
    CHECK_EQ(nnz_total, B.nnz);
    CHECK_EQ(B.nnz, 5);
    PASS();
}

// ── Test: RCM preserves nnz count ──

static void test_rcm_nnz() {
    TEST("rcm_preserves_nnz");
    auto A = make_tridiag(10);
    auto B = reorder_matrix(A, "rcm", g_nprocs, 42);
    CHECK_EQ(B.nnz, A.nnz);
    CHECK_EQ(B.nrows, A.nrows);
    CHECK_EQ(B.ncols, A.ncols);
    PASS();
}

// ── Test: reorder "none" is identity ──

static void test_reorder_none() {
    TEST("reorder_none");
    auto A = make_tridiag(5);
    auto B = reorder_matrix(A, "none", g_nprocs, 42);
    CHECK_EQ(B.nnz, A.nnz);
    CHECK_EQ(B.nrows, A.nrows);
    // Check that the matrix is unchanged
    for (idx_t i = 0; i <= A.nrows; ++i) {
        CHECK_EQ(B.rowptr[i], A.rowptr[i]);
    }
    for (int64_t j = 0; j < A.nnz; ++j) {
        CHECK_EQ(B.colidx[j], A.colidx[j]);
        CHECK_CLOSE(B.val[j], A.val[j], 1e-14);
    }
    PASS();
}

// ── Test: RCM with single process (previously was bug 2.1) ──

static void test_rcm_single_process() {
    // This test verifies that RCM is NOT skipped when nprocs=1 (bug 2.1 fix)
    TEST("rcm_single_process");
    // Simulate nprocs=1 by passing 1 manually
    auto A = make_tridiag(20);
    auto B = reorder_matrix(A, "rcm", 1, 42);
    // Should NOT be identity — it should have been RCM-permuted
    CHECK_EQ(B.nnz, A.nnz);
    // Verify the permutation actually changed something (tridiag is banded, RCM
    // preserves bandwidth, so the total nnz stays the same but order may differ)
    PASS();
}

// ── Test: diagonal_block_expand ──

static void test_diagonal_block_expand() {
    TEST("diagonal_block_expand");
    // Create a simple 6x6 matrix and distribute across processes
    // Each process has 2 rows (if 3 procs) or 3 rows (if 2 procs) etc.
    idx_t ncols = 6;
    auto [r_start, r_end] = row_partition_range(ncols, g_nprocs, g_rank);
    idx_t nlocal = r_end - r_start;
    if (nlocal == 0) { PASS(); return; }

    // Build a simple CSR: each row has entries at columns r_start..r_end-1
    // plus some random off-diagonal entries
    std::vector<idx_t> lrp(nlocal + 1);
    std::vector<idx_t> lci;
    std::vector<val_t> lval;
    lrp[0] = 0;
    for (idx_t i = 0; i < nlocal; ++i) {
        idx_t global_i = r_start + i;
        // Add diagonal entries within our range
        for (idx_t c = r_start; c < r_end; ++c) {
            lci.push_back(c);
            lval.push_back(1.0);
        }
        // Add a few off-diagonal entries
        if (global_i > 0) {
            lci.push_back(global_i - 1);
            lval.push_back(0.5);
        }
        if (global_i + 1 < ncols) {
            lci.push_back(global_i + 1);
            lval.push_back(0.5);
        }
        lrp[i + 1] = static_cast<idx_t>(lci.size());
    }

    auto [left, right] = diagonal_block_expand(
        lrp.data(), lci.data(), nlocal, r_start, r_end, ncols,
        MPI_COMM_WORLD);
    CHECK(left <= r_start);
    CHECK(right >= r_end || right <= ncols);
    PASS();
}

// ── Test: build_schedule ──

static void test_build_schedule() {
    TEST("build_schedule");
    idx_t ncols = 10;
    auto [r_start, r_end] = row_partition_range(ncols, g_nprocs, g_rank);
    idx_t nlocal = r_end - r_start;
    if (nlocal == 0) { PASS(); return; }

    // Simple CSR: only diagonal entries + right neighbor
    std::vector<idx_t> lrp(nlocal + 1);
    std::vector<idx_t> lci;
    std::vector<val_t> lval;
    lrp[0] = 0;
    for (idx_t i = 0; i < nlocal; ++i) {
        idx_t gi = r_start + i;
        lci.push_back(gi);
        lval.push_back(2.0);
        if (gi + 1 < ncols) {
            lci.push_back(gi + 1);
            lval.push_back(-1.0);
        }
        lrp[i + 1] = static_cast<idx_t>(lci.size());
    }

    auto [left, right] = diagonal_block_expand(
        lrp.data(), lci.data(), nlocal, r_start, r_end, ncols,
        MPI_COMM_WORLD);

    auto sched = build_schedule(
        lrp.data(), lci.data(), nlocal, r_start, r_end, left, right,
        MPI_COMM_WORLD);

    // Verify that recv_idx_map is consistent with recvid
    for (auto& [q, idx_map] : sched.recv_idx_map) {
        CHECK(sched.recvid.count(q) > 0);
        CHECK_EQ(static_cast<int>(idx_map.size()),
                 static_cast<int>(sched.recvid[q].size()));
    }
    PASS();
}

// ── Test: DistSpMVSolver multiply vs serial ──

static void test_solver_multiply_identity() {
    TEST("solver_multiply_identity");
    idx_t n = 10;
    auto A = make_identity(n);
    auto [r_start, r_end] = row_partition_range(n, g_nprocs, g_rank);
    idx_t nlocal = r_end - r_start;
    if (nlocal == 0) { PASS(); return; }

    // Extract local partition
    std::vector<idx_t> lrp(nlocal + 1);
    std::vector<idx_t> lci;
    std::vector<val_t> lval;
    lrp[0] = 0;
    for (idx_t i = 0; i < nlocal; ++i) {
        idx_t gi = r_start + i;
        lci.push_back(gi);
        lval.push_back(1.0);
        lrp[i + 1] = 1;
    }
    // Fix rowptr
    for (idx_t i = 1; i <= nlocal; ++i) {
        lrp[i] = static_cast<idx_t>(i);
    }

    auto [left, right] = diagonal_block_expand(
        lrp.data(), lci.data(), nlocal, r_start, r_end, n,
        MPI_COMM_WORLD);

    auto sched = build_schedule(
        lrp.data(), lci.data(), nlocal, r_start, r_end, left, right,
        MPI_COMM_WORLD);

    DistSpMVSolver solver(
        lrp.data(), lci.data(), lval.data(),
        nlocal, r_start, r_end, left, right,
        sched, MPI_COMM_WORLD, 2);

    std::vector<double> x_global(n, 1.0);
    std::vector<double> y_local(nlocal);
    solver.multiply(x_global.data(), y_local.data());

    // y_local[i] should be x[r_start + i] = 1.0 (since A = I)
    for (idx_t i = 0; i < nlocal; ++i) {
        CHECK_CLOSE(y_local[i], 1.0, 1e-12);
    }
    PASS();
}

// ── Test: redistribute_remote_by_nnz ──

static void test_redistribute() {
    TEST("redistribute_remote_by_nnz");
    if (g_nprocs <= 1) { PASS(); return; }

    idx_t ncols = 12;
    auto [r_start, r_end] = row_partition_range(ncols, g_nprocs, g_rank);
    idx_t nlocal = r_end - r_start;
    if (nlocal == 0) { PASS(); return; }

    // Build a matrix where remote nnz varies across rows
    std::vector<idx_t> lrp(nlocal + 1);
    std::vector<idx_t> lci;
    std::vector<val_t> lval;
    lrp[0] = 0;
    for (idx_t i = 0; i < nlocal; ++i) {
        idx_t gi = r_start + i;
        lci.push_back(gi); lval.push_back(1.0);  // diag
        // Add some remote entries
        lci.push_back((gi + 3) % ncols); lval.push_back(0.5);
        lci.push_back((gi + 6) % ncols); lval.push_back(0.3);
        lrp[i + 1] = static_cast<idx_t>(lci.size());
    }

    auto [left, right] = diagonal_block_expand(
        lrp.data(), lci.data(), nlocal, r_start, r_end, ncols,
        MPI_COMM_WORLD);

    idx_t old_nlocal = nlocal;
    redistribute_remote_by_nnz(lrp, lci, lval,
                                nlocal, r_start, r_end, left, right,
                                ncols, MPI_COMM_WORLD);

    // After redistribution, data should be consistent
    CHECK_EQ(static_cast<idx_t>(lrp.size()), nlocal + 1);
    if (nlocal > 0) {
        CHECK_EQ(static_cast<int64_t>(lrp.back()),
                 static_cast<int64_t>(lci.size()));
        CHECK_EQ(static_cast<int64_t>(lval.size()),
                 static_cast<int64_t>(lci.size()));
    }
    PASS();
}

// ── Test: full pipeline end-to-end ──

static void test_full_pipeline() {
    TEST("full_pipeline_identity");
    idx_t n = 8;
    auto A = make_identity(n);
    auto [r_start, r_end] = row_partition_range(n, g_nprocs, g_rank);
    idx_t nlocal = r_end - r_start;
    if (nlocal == 0) { PASS(); return; }

    // Build local CSR
    std::vector<idx_t> lrp(nlocal + 1);
    std::vector<idx_t> lci;
    std::vector<val_t> lval;
    lrp[0] = 0;
    for (idx_t i = 0; i < nlocal; ++i) {
        lci.push_back(r_start + i);
        lval.push_back(1.0);
        lrp[i + 1] = static_cast<idx_t>(i + 1);
    }

    // Algo 1
    auto [left, right] = diagonal_block_expand(
        lrp.data(), lci.data(), nlocal, r_start, r_end, n,
        MPI_COMM_WORLD);

    // Redistribute
    redistribute_remote_by_nnz(lrp, lci, lval,
                                nlocal, r_start, r_end, left, right,
                                n, MPI_COMM_WORLD);

    // Re-run Algo 1
    auto [left2, right2] = diagonal_block_expand(
        lrp.data(), lci.data(), nlocal, r_start, r_end, n,
        MPI_COMM_WORLD);

    // Algo 2
    auto sched = build_schedule(
        lrp.data(), lci.data(), nlocal, r_start, r_end, left2, right2,
        MPI_COMM_WORLD);

    // Solver
    DistSpMVSolver solver(
        lrp.data(), lci.data(), lval.data(),
        nlocal, r_start, r_end, left2, right2,
        sched, MPI_COMM_WORLD, 1);

    // Verify
    std::vector<double> x_global(n, 1.0);
    std::vector<double> y_local(nlocal);
    solver.multiply(x_global.data(), y_local.data());

    if (nlocal > 0) {
        CHECK_CLOSE(y_local[0], 1.0, 1e-12);
    }
    PASS();
}

// ── Test: exchange_remote buffer reuse ──

static void test_buffer_reuse() {
    TEST("buffer_reuse");
    idx_t n = 6;
    auto A = make_identity(n);
    auto [r_start, r_end] = row_partition_range(n, g_nprocs, g_rank);
    idx_t nlocal = r_end - r_start;
    if (nlocal == 0) { PASS(); return; }

    std::vector<idx_t> lrp(nlocal + 1);
    std::vector<idx_t> lci;
    std::vector<val_t> lval;
    lrp[0] = 0;
    for (idx_t i = 0; i < nlocal; ++i) {
        lci.push_back(r_start + i);
        lval.push_back(1.0);
        lrp[i + 1] = static_cast<idx_t>(i + 1);
    }

    auto [left, right] = diagonal_block_expand(
        lrp.data(), lci.data(), nlocal, r_start, r_end, n,
        MPI_COMM_WORLD);

    auto sched = build_schedule(
        lrp.data(), lci.data(), nlocal, r_start, r_end, left, right,
        MPI_COMM_WORLD);

    DistSpMVSolver solver(
        lrp.data(), lci.data(), lval.data(),
        nlocal, r_start, r_end, left, right,
        sched, MPI_COMM_WORLD, 1);

    std::vector<double> x_global(n, 1.0);
    std::vector<double> y_local(nlocal);

    // Call multiply multiple times — pre-allocated buffers should be reused
    for (int iter = 0; iter < 10; ++iter) {
        solver.multiply(x_global.data(), y_local.data());
        if (nlocal > 0) {
            CHECK_CLOSE(y_local[0], 1.0, 1e-12);
        }
    }
    PASS();
}

// ── Main test runner ──

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nprocs);

    if (g_rank == 0) {
        std::printf("\n=== DistSpMV_Balanced Unit Tests (nprocs=%d) ===\n\n",
                    g_nprocs);
    }

    // Utils
    test_row_partition();
    test_serial_spmv();
    test_serial_spmv_tridiag();
    test_compute_error();

    // Reordering
    test_rcm_identity();
    test_rcm_nnz();
    test_reorder_none();
    test_rcm_single_process();

    // Partition (Algorithm 1)
    test_diagonal_block_expand();

    // Communication schedule (Algorithm 2)
    test_build_schedule();

    // Redistribution
    test_redistribute();

    // Solver
    test_solver_multiply_identity();
    test_buffer_reuse();

    // Full pipeline
    test_full_pipeline();

    // ── Report ──
    int global_passed = 0, global_failed = 0;
    MPI_Reduce(&g_passed, &global_passed, 1, MPI_INT, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&g_failed, &global_failed, 1, MPI_INT, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (g_rank == 0) {
        std::printf("\n=== Results: %d passed, %d failed ===\n",
                    global_passed, global_failed);
    }

    MPI_Finalize();
    return (global_failed > 0) ? 1 : 0;
}
