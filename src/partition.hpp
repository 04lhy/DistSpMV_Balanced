#pragma once
/**
 * Algorithm 1: Diagonal Block Column Boundary Expansion.
 *
 * Paper reference: Section III.B, Algorithm 1, Fig. 3.
 *
 * Determines [left, right) column range so that each local row has at
 * least 'lower_bound' nonzeros inside the diagonal block.
 */

#include <mpi.h>

#include <utility>

#include "types.hpp"

namespace distspmv {

/// Run Algorithm 1.  Returns (left, right) boundaries.
///
/// rowptr length = nlocal + 1, colidx length = nnz_local
/// r_start, r_end define this rank's global row range.
std::pair<idx_t, idx_t> diagonal_block_expand(
    const idx_t* rowptr, const idx_t* colidx,
    idx_t nlocal, idx_t r_start, idx_t r_end, idx_t ncols,
    MPI_Comm comm);

}  // namespace distspmv
