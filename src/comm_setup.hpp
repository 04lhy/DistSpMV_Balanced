#pragma once
/**
 * Algorithm 2: Communication Information Collection.
 *
 * Paper reference: Section III.C, Algorithm 2.
 *
 * Builds the communication schedule: which columns each process sends
 * to / receives from every other process.
 */

#include <mpi.h>

#include "types.hpp"

namespace distspmv {

/// Build communication schedule.  Returns sendid, recvid, recv_idx_map.
///
/// Column ownership follows the initial row distribution: rank q owns
/// columns [all_r_start[q], all_r_end[q]).
CommSchedule build_schedule(
    const idx_t* rowptr, const idx_t* colidx,
    idx_t nlocal, idx_t r_start, idx_t r_end,
    idx_t left, idx_t right, MPI_Comm comm);

}  // namespace distspmv
