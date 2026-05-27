#pragma once
/**
 * Common type definitions for DistSpMV_Balanced.
 *
 * Convention: 0-based indexing throughout.  All CSR arrays use int32 for
 * indices and float64 for values, matching the original Python code.
 */

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace distspmv {

using idx_t = int32_t;
using val_t = double;

// ── CSR (Compressed Sparse Row) storage ──────────────────────────────

struct CSRMatrix {
    std::vector<idx_t> rowptr;  // length nrows + 1
    std::vector<idx_t> colidx;  // length nnz
    std::vector<val_t> val;     // length nnz
    idx_t nrows = 0;
    idx_t ncols = 0;
    int64_t nnz = 0;
};

// ── Communication schedule (output of Algorithm 2) ───────────────────

struct CommSchedule {
    /// Columns to send to each peer process.
    std::unordered_map<int, std::vector<idx_t>> sendid;
    /// Columns to receive from each peer process.
    std::unordered_map<int, std::vector<idx_t>> recvid;
    /// Maps global column → local buffer position for each peer.
    std::unordered_map<int, std::unordered_map<idx_t, int>> recv_idx_map;
};

}  // namespace distspmv
