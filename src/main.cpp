/**
 * DistSpMV_Balanced — Main entry point.
 *
 * Orchestrates the full distributed SpMV pipeline:
 *   1. Read MTX (rank 0), broadcast CSR to all ranks.
 *   2. (Optional) Reorder matrix via RCM / METIS / none.
 *   3. Distribute rows across MPI processes (contiguous 1-D).
 *   4. Algorithm 1 — diagonal block boundary expansion.
 *   5. Algorithm 2 — communication schedule construction.
 *   6. Algorithms 3+4 — repeated SpMV benchmarking.
 *   7. Correctness verification.
 *
 * Usage:
 *   mpiexec -n 4 ./build/dist_spmv --matrix data/cant.mtx --threads 4
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

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

// ── Simple CLI parser ────────────────────────────────────────────────

struct Args {
    std::string matrix;
    int threads = 1;
    std::string reorder = "rcm";
    int benchmark = 50;
    int warmup = 5;
    int seed = 42;
    std::string output;
    bool no_verify = false;
    bool verbose = false;
};

void print_usage(const char* prog) {
    std::printf(
        "Usage: %s --matrix PATH [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  --matrix PATH      Path to .mtx file (required)\n"
        "  --threads N        Threads per rank for local SpMV (default: 1)\n"
        "  --reorder METHOD   rcm | metis | none (default: rcm)\n"
        "  --benchmark N      Number of timed SpMV repetitions (default: 50)\n"
        "  --warmup N         Number of warmup SpMV calls (default: 5)\n"
        "  --seed N           Random seed (default: 42)\n"
        "  --output PATH      JSON output path for metrics (optional)\n"
        "  --no-verify        Skip correctness verification\n"
        "  --verbose          Enable verbose logging\n"
        "  --help             Show this message\n",
        prog);
}

Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--matrix" && i + 1 < argc) {
            args.matrix = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            args.threads = std::atoi(argv[++i]);
        } else if (arg == "--reorder" && i + 1 < argc) {
            args.reorder = argv[++i];
        } else if (arg == "--benchmark" && i + 1 < argc) {
            args.benchmark = std::atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            args.warmup = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            args.seed = std::atoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            args.output = argv[++i];
        } else if (arg == "--no-verify") {
            args.no_verify = true;
        } else if (arg == "--verbose") {
            args.verbose = true;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    if (args.matrix.empty()) {
        std::fprintf(stderr, "ERROR: --matrix is required\n");
        print_usage(argv[0]);
        std::exit(1);
    }
    return args;
}

// ── Main pipeline ────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Parse args (all ranks need the values; rank 0 parses, others bcast)
    Args args;
    int threads = 0, benchmark = 0, warmup = 0, seed = 0;
    bool no_verify = false;

    if (rank == 0) {
        args = parse_args(argc, argv);
        threads = args.threads;
        benchmark = args.benchmark;
        warmup = args.warmup;
        seed = args.seed;
        no_verify = args.no_verify;
    }

    MPI_Bcast(&threads, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&benchmark, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&warmup, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int no_verify_int = no_verify ? 1 : 0;
    MPI_Bcast(&no_verify_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    no_verify = (no_verify_int != 0);

    double t_total_start = MPI_Wtime();

    // ── Step 1: Load matrix ──
    CSRMatrix A_global;
    A_global.nrows = 0;
    A_global.ncols = 0;
    A_global.nnz = 0;

    double t_reorder = 0.0;

    if (rank == 0) {
        A_global = read_mtx(args.matrix);
        log_info(rank, "Global matrix: %d x %d, %lld nnz",
                 A_global.nrows, A_global.ncols, A_global.nnz);

        // ── Step 2: Reordering ──
        double t0 = MPI_Wtime();
        A_global = reorder_matrix(A_global, args.reorder, nprocs, seed);
        t_reorder = MPI_Wtime() - t0;
    }

    // Broadcast dimensions
    idx_t nrows_g = A_global.nrows;
    idx_t ncols_g = A_global.ncols;
    int64_t nnz_g = A_global.nnz;
    int reorder_len = static_cast<int>(args.reorder.size());

    MPI_Bcast(&nrows_g, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ncols_g, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz_g, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&reorder_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&t_reorder, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast reorder method string
    std::vector<char> reorder_buf(reorder_len + 1);
    if (rank == 0) {
        std::strncpy(reorder_buf.data(), args.reorder.c_str(), reorder_len + 1);
    }
    MPI_Bcast(reorder_buf.data(), reorder_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Broadcast CSR arrays
    std::vector<idx_t> rowptr_g(nrows_g + 1);
    std::vector<idx_t> colidx_g(nnz_g);
    std::vector<val_t> val_g(nnz_g);

    if (rank == 0) {
        std::copy(A_global.rowptr.begin(), A_global.rowptr.end(),
                  rowptr_g.begin());
        std::copy(A_global.colidx.begin(), A_global.colidx.end(),
                  colidx_g.begin());
        std::copy(A_global.val.begin(), A_global.val.end(), val_g.begin());
    }

    MPI_Bcast(rowptr_g.data(), nrows_g + 1, MPI_INT, 0, MPI_COMM_WORLD);
    assert(nnz_g <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
           "nnz exceeds MPI int range");
    MPI_Bcast(colidx_g.data(), static_cast<int>(nnz_g), MPI_INT,
              0, MPI_COMM_WORLD);
    MPI_Bcast(val_g.data(), static_cast<int>(nnz_g), MPI_DOUBLE,
              0, MPI_COMM_WORLD);

    // ── Step 3: Row-wise distribution ──
    auto [r_start, r_end] = row_partition_range(nrows_g, nprocs, rank);
    idx_t nlocal = r_end - r_start;

    idx_t local_nnz = rowptr_g[r_end] - rowptr_g[r_start];
    std::vector<idx_t> local_rowptr(nlocal + 1);
    for (idx_t i = 0; i <= nlocal; ++i) {
        local_rowptr[i] = rowptr_g[r_start + i] - rowptr_g[r_start];
    }

    std::vector<idx_t> local_colidx(colidx_g.begin() + rowptr_g[r_start],
                                     colidx_g.begin() + rowptr_g[r_start]
                                     + local_nnz);
    std::vector<val_t> local_val(val_g.begin() + rowptr_g[r_start],
                                  val_g.begin() + rowptr_g[r_start]
                                  + local_nnz);

    log_info(rank, "Local partition: rows [%d, %d) nlocal=%d nnz=%d",
             r_start, r_end, nlocal, local_nnz);

    // ── Step 4: Algorithm 1 — diagonal block expansion ──
    double t_algo1_start = MPI_Wtime();
    auto [left, right] = diagonal_block_expand(
        local_rowptr.data(), local_colidx.data(),
        nlocal, r_start, r_end, ncols_g, MPI_COMM_WORLD);
    double t_algo1 = MPI_Wtime() - t_algo1_start;

    // ── Step 4.5: Redistribute remote matrix by nnz (paper core strategy) ──
    double t_redis_start = MPI_Wtime();
    redistribute_remote_by_nnz(
        local_rowptr, local_colidx, local_val,
        nlocal, r_start, r_end, left, right,
        ncols_g, MPI_COMM_WORLD);

    // Re-run Algorithm 1 on redistributed data
    auto [left2, right2] = diagonal_block_expand(
        local_rowptr.data(), local_colidx.data(),
        nlocal, r_start, r_end, ncols_g, MPI_COMM_WORLD);
    left = left2;
    right = right2;
    double t_algo1_redis = MPI_Wtime() - t_redis_start;  // includes redistribution + re-Algo1

    // ── Step 5: Algorithm 2 — communication schedule ──
    double t_algo2_start = MPI_Wtime();
    auto sched = build_schedule(
        local_rowptr.data(), local_colidx.data(),
        nlocal, r_start, r_end, left, right, MPI_COMM_WORLD);
    double t_algo2 = MPI_Wtime() - t_algo2_start;

    // ── Step 6: Build solver ──
    DistSpMVSolver solver(
        local_rowptr.data(), local_colidx.data(), local_val.data(),
        nlocal, r_start, r_end, left, right,
        sched, MPI_COMM_WORLD, threads);

    // ── Step 7: Prepare x vector ──
    std::vector<double> x_global(ncols_g);
    if (rank == 0) {
        std::mt19937_64 rng(static_cast<uint64_t>(seed));
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (idx_t i = 0; i < ncols_g; ++i) {
            x_global[i] = dist(rng);
        }
    }
    MPI_Bcast(x_global.data(), ncols_g, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ── Step 8: Benchmark ──
    double t_preprocess = MPI_Wtime() - t_total_start;

    auto [avg_time, std_time] = solver.multiply_benchmark(
        x_global.data(), warmup, benchmark);

    double gflops_val = gflops(nnz_g, avg_time);

    // ── Step 9: Correctness verification ──
    bool is_correct = true;
    double error = 0.0;

    if (!no_verify) {
        std::vector<double> y_local(nlocal);
        solver.multiply(x_global.data(), y_local.data());

        // Gather all y_local pieces to rank 0
        std::vector<int> recv_counts(nprocs);
        int nlocal_int = static_cast<int>(nlocal);
        MPI_Gather(&nlocal_int, 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                   0, MPI_COMM_WORLD);

        std::vector<double> y_gathered;
        std::vector<int> displs(nprocs, 0);
        int total_local = 0;

        if (rank == 0) {
            for (int q = 0; q < nprocs; ++q) {
                displs[q] = total_local;
                total_local += recv_counts[q];
            }
            y_gathered.resize(total_local);
        }

        MPI_Gatherv(y_local.data(), nlocal_int, MPI_DOUBLE,
                    y_gathered.data(), recv_counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Compute reference: y_ref = A @ x (using global CSR on rank 0)
            std::vector<double> y_ref(nrows_g);
            // Build temporary CSRMatrix for the global (reordered) matrix
            CSRMatrix A_tmp;
            A_tmp.nrows = nrows_g;
            A_tmp.ncols = ncols_g;
            A_tmp.nnz = nnz_g;
            A_tmp.rowptr = rowptr_g;
            A_tmp.colidx = colidx_g;
            A_tmp.val = val_g;
            serial_spmv(A_tmp, x_global.data(), y_ref.data());

            error = compute_error(y_gathered.data(), y_ref.data(), nrows_g);
            is_correct = (error < 1e-6);

            log_info(rank, "Correctness: %s (relative error = %.2e)",
                     is_correct ? "PASS" : "FAIL", error);
        }

        int correct_int = is_correct ? 1 : 0;
        MPI_Bcast(&correct_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        is_correct = (correct_int != 0);
        MPI_Bcast(&error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // ── Step 10: Report ──
    if (rank == 0) {
        int64_t comm_vol_recv = 0, comm_vol_send = 0;
        for (auto& [_, cols] : sched.recvid) comm_vol_recv += cols.size();
        for (auto& [_, cols] : sched.sendid) comm_vol_send += cols.size();

        log_info(rank, "Results: avg_time=%.6f s  GFlops=%.4f  "
                 "diag_nnz=%d  offdiag_nnz=%d",
                 avg_time, gflops_val, solver.nnz_diag(), solver.nnz_offdiag());

        // JSON output
        if (!args.output.empty()) {
            std::string outdir = args.output;
            // Extract directory
            auto slash_pos = outdir.find_last_of("/\\");
            if (slash_pos != std::string::npos) {
                std::string dir = outdir.substr(0, slash_pos);
                // Best-effort directory creation — ignore errors (dir may
                // already exist or lack permissions).
#ifdef _WIN32
                _mkdir(dir.c_str());
#else
                mkdir(dir.c_str(), 0755);
#endif
            }

            FILE* fp = std::fopen(args.output.c_str(), "w");
            if (fp) {
                std::fprintf(fp, "{\n");
                std::fprintf(fp, "  \"matrix\": \"%s\",\n",
                             args.matrix.c_str());
                std::fprintf(fp, "  \"nrows\": %d,\n", nrows_g);
                std::fprintf(fp, "  \"ncols\": %d,\n", ncols_g);
                std::fprintf(fp, "  \"nnz_global\": %lld,\n", nnz_g);
                std::fprintf(fp, "  \"nprocs\": %d,\n", nprocs);
                std::fprintf(fp, "  \"n_threads\": %d,\n", threads);
                std::fprintf(fp, "  \"reorder\": \"%s\",\n",
                             reorder_buf.data());
                std::fprintf(fp, "  \"correct\": %s,\n",
                             is_correct ? "true" : "false");
                std::fprintf(fp, "  \"error\": %.10e,\n", error);
                std::fprintf(fp, "  \"preprocess_time_s\": %.6f,\n",
                             t_preprocess);
                std::fprintf(fp, "  \"reorder_time_s\": %.6f,\n", t_reorder);
                std::fprintf(fp, "  \"algo1_time_s\": %.6f,\n", t_algo1);
                std::fprintf(fp, "  \"redistribute_time_s\": %.6f,\n",
                             t_algo1_redis);
                std::fprintf(fp, "  \"algo2_time_s\": %.6f,\n", t_algo2);
                std::fprintf(fp, "  \"avg_spmv_time_s\": %.6f,\n", avg_time);
                std::fprintf(fp, "  \"std_spmv_time_s\": %.6f,\n", std_time);
                std::fprintf(fp, "  \"gflops\": %.6f,\n", gflops_val);
                std::fprintf(fp, "  \"diag_boundary_left\": %d,\n", left);
                std::fprintf(fp, "  \"diag_boundary_right\": %d,\n", right);
                std::fprintf(fp, "  \"nnz_local\": %d,\n", local_nnz);
                std::fprintf(fp, "  \"nnz_diag\": %d,\n", solver.nnz_diag());
                std::fprintf(fp, "  \"nnz_offdiag\": %d,\n",
                             solver.nnz_offdiag());
                std::fprintf(fp, "  \"comm_volume_recv\": %lld,\n",
                             comm_vol_recv);
                std::fprintf(fp, "  \"comm_volume_send\": %lld\n",
                             comm_vol_send);
                std::fprintf(fp, "}\n");
                std::fclose(fp);
                log_info(rank, "Metrics written to %s",
                         args.output.c_str());
            } else {
                std::fprintf(stderr, "[ERROR] Cannot write to %s\n",
                             args.output.c_str());
            }
        }
    }

    MPI_Finalize();
    return 0;
}
