#pragma once
/**
 * Matrix reordering for improved locality.
 *
 * Supported methods:
 *   "none" — identity permutation
 *   "rcm"  — Reverse Cuthill-McKee (reduces bandwidth)
 *   "metis"— METIS partitioning (requires DISTSPMV_HAS_METIS)
 */

#include <string>
#include <vector>

#include "types.hpp"

namespace distspmv {

/// Apply symmetric reordering P·A·P^T.  Returns the permuted matrix and
/// the permutation array perm[i] = new index of original row i.
CSRMatrix reorder_matrix(const CSRMatrix& A, const std::string& method,
                         int nparts = 1, int seed = 42);

}  // namespace distspmv
