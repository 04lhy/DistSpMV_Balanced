#include "partition.hpp"

#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace distspmv {

std::pair<idx_t, idx_t> diagonal_block_expand(
    const idx_t* rowptr, const idx_t* colidx,
    idx_t nlocal, idx_t r_start, idx_t r_end, idx_t ncols,
    MPI_Comm comm) {

    if (nlocal == 0) return {0, 0};

    // ── S1: Initialize boundaries & count per-row ──
    idx_t left = r_start;
    idx_t right = std::min(r_end, ncols);

    std::vector<idx_t> diag(nlocal, 0);
    std::vector<idx_t> off_left(nlocal, 0);
    std::vector<idx_t> off_right(nlocal, 0);
    std::vector<idx_t> total_nnz(nlocal, 0);

    for (idx_t i = 0; i < nlocal; ++i) {
        total_nnz[i] = rowptr[i + 1] - rowptr[i];
        for (idx_t j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            idx_t c = colidx[j];
            if (c < left)       off_left[i]++;
            else if (c >= right) off_right[i]++;
            else                 diag[i]++;
        }
    }

    // ── S2: Compute lower_bound ──
    idx_t local_max = 0;
    for (idx_t i = 0; i < nlocal; ++i) {
        if (diag[i] > local_max) local_max = diag[i];
    }

    idx_t lower_bound = 0;
    MPI_Allreduce(&local_max, &lower_bound, 1, MPI_INT, MPI_MAX, comm);
    if (lower_bound > ncols) lower_bound = ncols;

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        std::printf("[rank %2d] Algo1 init: left=%d right=%d local_max_diag=%d "
                    "lower_bound=%d\n",
                    rank, left, right, local_max, lower_bound);
    }

    // ── S3: Build column-to-rows index for fast lookup ──
    // Only for columns that appear in this local partition
    std::unordered_map<idx_t, std::vector<idx_t>> col_to_rows;
    for (idx_t i = 0; i < nlocal; ++i) {
        for (idx_t j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            idx_t c = colidx[j];
            col_to_rows[c].push_back(i);
        }
    }

    // ── S4: Greedy boundary expansion ──
    idx_t max_iter = ncols;
    for (idx_t iter = 0; iter < max_iter; ++iter) {
        // Find deficient rows
        std::unordered_set<idx_t> deficient;
        std::vector<idx_t> target(nlocal);
        for (idx_t i = 0; i < nlocal; ++i) {
            target[i] = std::min(lower_bound, total_nnz[i]);
            if (diag[i] < target[i]) {
                deficient.insert(i);
            }
        }

        if (deficient.empty()) {
            if (rank == 0) {
                std::printf("[rank %2d] Algo1 converged after %d iterations\n",
                            rank, iter);
            }
            break;
        }

        // Count gains
        idx_t left_gain = 0;
        if (left > 0) {
            auto it = col_to_rows.find(left - 1);
            if (it != col_to_rows.end()) {
                for (idx_t r : it->second) {
                    if (deficient.count(r)) left_gain++;
                }
            }
        }

        idx_t right_gain = 0;
        if (right < ncols) {
            auto it = col_to_rows.find(right);
            if (it != col_to_rows.end()) {
                for (idx_t r : it->second) {
                    if (deficient.count(r)) right_gain++;
                }
            }
        }

        if (left_gain == 0 && right_gain == 0) {
            if (rank == 0) {
                std::printf("[rank %2d] Algo1 stalled at iter %d: %zu rows "
                            "deficient\n",
                            rank, iter, deficient.size());
            }
            break;
        }

        // Expand toward larger gain
        if (left_gain >= right_gain && left > 0) {
            left--;
            auto it = col_to_rows.find(left);
            if (it != col_to_rows.end()) {
                for (idx_t r : it->second) {
                    off_left[r]--;
                    diag[r]++;
                }
            }
        } else if (right < ncols) {
            auto it = col_to_rows.find(right);
            if (it != col_to_rows.end()) {
                for (idx_t r : it->second) {
                    off_right[r]--;
                    diag[r]++;
                }
            }
            right++;
        } else {
            break;
        }
    }

    if (rank == 0) {
        std::printf("[rank %2d] Algo1 result: left=%d right=%d "
                    "(initial [%d,%d))\n",
                    rank, left, right, r_start, std::min(r_end, ncols));
    }

    return {left, right};
}

}  // namespace distspmv
