#pragma once
/**
 * Remote matrix redistribution by non-zero count.
 *
 * Paper reference: Section III.B (after Algorithm 1, before Algorithm 2).
 *
 * The remote matrix (entries outside [left, right)) is redistributed so
 * that each process handles roughly equal remote nnz, not equal row count.
 */

#include <mpi.h>

#include "types.hpp"

namespace distspmv {

/// Redistribute rows so that each process has roughly equal total remote nnz.
/// On return, rowptr/colidx/val are updated, and nlocal/r_start/r_end/left/right
/// reflect the new distribution.  reorder_len is the length of the reorder
/// method name string (used for logging context).
void redistribute_remote_by_nnz(
    std::vector<idx_t>& rowptr, std::vector<idx_t>& colidx,
    std::vector<val_t>& val,
    idx_t& nlocal, idx_t& r_start, idx_t& r_end,
    idx_t& left, idx_t& right,
    idx_t ncols, MPI_Comm comm);

}  // namespace distspmv
