#include "redistribute.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <numeric>
#include <vector>

namespace distspmv {

void redistribute_remote_by_nnz(
    std::vector<idx_t>& rowptr, std::vector<idx_t>& colidx,
    std::vector<val_t>& val,
    idx_t& nlocal, idx_t& r_start, idx_t& r_end,
    idx_t& left, idx_t& right,
    idx_t ncols, MPI_Comm comm)
{
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    if (nprocs <= 1 || nlocal == 0) return;

    // ── Step 1: Count remote nnz per row ──
    std::vector<int64_t> remote_nnz(nlocal, 0);
    int64_t local_remote_total = 0;
    for (idx_t i = 0; i < nlocal; ++i) {
        int64_t cnt = 0;
        for (idx_t j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            idx_t c = colidx[j];
            if (c < left || c >= right) cnt++;
        }
        remote_nnz[i] = cnt;
        local_remote_total += cnt;
    }

    // ── Step 2: Allgather remote totals ──
    std::vector<int64_t> all_remote_totals(nprocs);
    MPI_Allgather(&local_remote_total, 1, MPI_LONG_LONG,
                  all_remote_totals.data(), 1, MPI_LONG_LONG, comm);

    int64_t global_remote_total = 0;
    for (int q = 0; q < nprocs; ++q) {
        global_remote_total += all_remote_totals[q];
    }

    if (global_remote_total == 0) {
        if (rank == 0) {
            std::printf("Redistribute: no remote nnz, skipping.\n");
        }
        return;
    }

    // ── Step 3: Build per-row global remote-nnz prefix ──
    int64_t my_prefix_offset = 0;
    for (int q = 0; q < rank; ++q) {
        my_prefix_offset += all_remote_totals[q];
    }

    std::vector<int64_t> local_prefix(nlocal + 1);
    local_prefix[0] = 0;
    for (idx_t i = 0; i < nlocal; ++i) {
        local_prefix[i + 1] = local_prefix[i] + remote_nnz[i];
    }

    // ── Step 4: Determine target remote-nnz range for each process ──
    int64_t target_per_proc = (global_remote_total + nprocs - 1) / nprocs;

    // Find global row boundaries that partition remote nnz evenly.
    // Each process q needs: sum of remote_nnz in its rows ≈ target_per_proc.
    //
    // For each process, find the global row index where cumulative remote nnz
    // crosses q * target_per_proc.  We do this via prefix-sum exchange.

    // Each process has a contiguous block of global rows [r_start, r_end).
    // We know the cumulative remote nnz at r_start (my_prefix_offset) and at
    // each row within the block.  We need to find the global row indices
    // at which cumulative nnz crosses multiples of target_per_proc.

    // ── Determine new contiguous row partition ──
    // Strategy: find split rows at q*target_per_proc boundaries.
    // The split row for boundary B = q*target_per_proc is the row where
    //   global_cumulative_nnz at row_start <= B < global_cumulative_nnz at row_end
    struct Split {
        int64_t split_nnz;  // the nnz value at which to split
        int rank;
    };
    std::vector<Split> splits;
    for (int q = 1; q < nprocs; ++q) {
        splits.push_back({q * target_per_proc, q});
    }

    // For each split, determine which process owns the split row
    // The process where cumulative_total at its r_start <= split_nnz < cumulative_total at its r_end
    std::vector<int64_t> cum_at_start(nprocs);
    int64_t acc = 0;
    for (int q = 0; q < nprocs; ++q) {
        cum_at_start[q] = acc;
        acc += all_remote_totals[q];
    }

    // For each split, find which process and which local row
    struct SplitResult {
        int src_rank;
        idx_t local_row_idx;  // the row AFTER which we split (0-based within src_rank)
        idx_t global_row;     // the global row index of the first row of the next process
    };
    std::vector<SplitResult> split_results(splits.size());

    for (std::size_t si = 0; si < splits.size(); ++si) {
        int64_t target_nnz = splits[si].split_nnz;
        // Find the process where this target falls
        int src = nprocs - 1;
        for (int q = 0; q < nprocs; ++q) {
            if (target_nnz < cum_at_start[q] + all_remote_totals[q]) {
                src = q;
                break;
            }
        }
        split_results[si].src_rank = src;
        // local_nnz_offset within source process
        int64_t local_offset = target_nnz - cum_at_start[src];
        // Default: split at beginning
        split_results[si].local_row_idx = 0;
        split_results[si].global_row = 0;
    }

    // Each process that owns a split computes the exact row
    // We need to communicate: for each split, the src process finds the row and broadcasts
    for (std::size_t si = 0; si < splits.size(); ++si) {
        int src = split_results[si].src_rank;
        int64_t local_offset = splits[si].split_nnz - cum_at_start[src];

        idx_t split_row = 0;
        idx_t global_split = 0;

        if (rank == src) {
            // Find the row where local_prefix[row] <= local_offset < local_prefix[row+1]
            // i.e., the first row after accumulating local_offset nnz
            for (idx_t i = 0; i < nlocal; ++i) {
                if (local_prefix[i + 1] > local_offset) {
                    split_row = i + 1;  // split AFTER row i
                    break;
                }
            }
            if (split_row == 0) split_row = nlocal;
            global_split = r_start + split_row;
        }

        // Broadcast the split row info to all processes
        int global_int = static_cast<int>(global_split);
        MPI_Bcast(&global_int, 1, MPI_INT, src, comm);
        split_results[si].global_row = static_cast<idx_t>(global_int);
    }

    // ── Step 5: Compute new contiguous row ranges ──
    std::vector<idx_t> new_r_starts(nprocs);
    std::vector<idx_t> new_r_ends(nprocs);

    new_r_starts[0] = 0;
    for (int q = 1; q < nprocs; ++q) {
        new_r_starts[q] = split_results[q - 1].global_row;
    }
    for (int q = 0; q < nprocs - 1; ++q) {
        new_r_ends[q] = new_r_starts[q + 1];
    }
    new_r_ends[nprocs - 1] = ncols;

    idx_t new_r_start = new_r_starts[rank];
    idx_t new_r_end = new_r_ends[rank];
    idx_t new_nlocal = new_r_end - new_r_start;

    // ── Step 6: Exchange rows to achieve contiguous distribution ──
    // Pack rows that belong to other processes (with their global row index)
    struct SendRow {
        idx_t gid;
        idx_t nnz;
        std::vector<idx_t> cols;
        std::vector<val_t> vals;
    };

    std::vector<std::vector<SendRow>> send_rows(nprocs);

    for (idx_t i = 0; i < nlocal; ++i) {
        idx_t gid = r_start + i;
        int dest = -1;
        for (int q = 0; q < nprocs; ++q) {
            if (new_r_starts[q] <= gid && gid < new_r_ends[q]) {
                dest = q;
                break;
            }
        }
        if (dest == -1) dest = nprocs - 1;

        if (dest != rank) {
            SendRow sr;
            sr.gid = gid;
            sr.nnz = rowptr[i + 1] - rowptr[i];
            sr.cols.assign(&colidx[rowptr[i]], &colidx[rowptr[i + 1]]);
            sr.vals.assign(&val[rowptr[i]], &val[rowptr[i + 1]]);
            send_rows[dest].push_back(std::move(sr));
        }
    }

    // Exchange sizes
    std::vector<int> send_counts(nprocs, 0);
    for (int q = 0; q < nprocs; ++q) {
        send_counts[q] = static_cast<int>(send_rows[q].size());
    }
    std::vector<int> recv_counts(nprocs, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

    // Serialize: format: [nrows] [gid0 nnz0 cols0... vals0... gid1 nnz1 cols1... vals1...]
    std::vector<std::vector<idx_t>> send_idx(nprocs);
    std::vector<std::vector<val_t>> send_val(nprocs);

    for (int q = 0; q < nprocs; ++q) {
        if (send_rows[q].empty()) continue;
        send_idx[q].push_back(static_cast<idx_t>(send_rows[q].size()));
        for (auto& sr : send_rows[q]) {
            send_idx[q].push_back(sr.gid);
            send_idx[q].push_back(sr.nnz);
            send_idx[q].insert(send_idx[q].end(), sr.cols.begin(), sr.cols.end());
            send_val[q].insert(send_val[q].end(), sr.vals.begin(), sr.vals.end());
        }
    }

    std::vector<int> send_sizes_idx(nprocs, 0), send_sizes_val(nprocs, 0);
    for (int q = 0; q < nprocs; ++q) {
        send_sizes_idx[q] = static_cast<int>(send_idx[q].size());
        send_sizes_val[q] = static_cast<int>(send_val[q].size());
    }
    std::vector<int> recv_sizes_idx(nprocs, 0), recv_sizes_val(nprocs, 0);
    MPI_Alltoall(send_sizes_idx.data(), 1, MPI_INT, recv_sizes_idx.data(), 1, MPI_INT, comm);
    MPI_Alltoall(send_sizes_val.data(), 1, MPI_INT, recv_sizes_val.data(), 1, MPI_INT, comm);

    std::vector<std::vector<idx_t>> recv_idx(nprocs);
    std::vector<std::vector<val_t>> recv_val(nprocs);
    for (int q = 0; q < nprocs; ++q) {
        if (recv_sizes_idx[q] > 0) recv_idx[q].resize(recv_sizes_idx[q]);
        if (recv_sizes_val[q] > 0) recv_val[q].resize(recv_sizes_val[q]);
    }

    std::vector<MPI_Request> reqs;
    for (int q = 0; q < nprocs; ++q) {
        if (recv_sizes_idx[q] > 0) {
            MPI_Request r;
            MPI_Irecv(recv_idx[q].data(), recv_sizes_idx[q], MPI_INT,
                      q, 300 + q, comm, &r);
            reqs.push_back(r);
        }
        if (recv_sizes_val[q] > 0) {
            MPI_Request r;
            MPI_Irecv(recv_val[q].data(), recv_sizes_val[q], MPI_DOUBLE,
                      q, 400 + q, comm, &r);
            reqs.push_back(r);
        }
    }
    for (int q = 0; q < nprocs; ++q) {
        if (!send_idx[q].empty()) {
            MPI_Request r;
            MPI_Isend(send_idx[q].data(), send_sizes_idx[q], MPI_INT,
                      q, 300 + rank, comm, &r);
            reqs.push_back(r);
        }
        if (!send_val[q].empty()) {
            MPI_Request r;
            MPI_Isend(send_val[q].data(), send_sizes_val[q], MPI_DOUBLE,
                      q, 400 + rank, comm, &r);
            reqs.push_back(r);
        }
    }
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

    // ── Step 7: Rebuild local CSR ──
    struct GidRow {
        idx_t gid;
        idx_t nnz;
        std::vector<idx_t> cols;
        std::vector<val_t> vals;
    };
    std::vector<GidRow> all_rows;

    // Kept rows (those in our new range)
    for (idx_t i = 0; i < nlocal; ++i) {
        idx_t gid = r_start + i;
        if (new_r_start <= gid && gid < new_r_end) {
            GidRow gr;
            gr.gid = gid;
            gr.nnz = rowptr[i + 1] - rowptr[i];
            gr.cols.assign(&colidx[rowptr[i]], &colidx[rowptr[i + 1]]);
            gr.vals.assign(&val[rowptr[i]], &val[rowptr[i + 1]]);
            all_rows.push_back(std::move(gr));
        }
    }

    // Received rows
    for (int q = 0; q < nprocs; ++q) {
        if (recv_idx[q].empty()) continue;
        const auto& bi = recv_idx[q];
        const auto& bv = recv_val[q];
        idx_t nr = bi[0];
        std::size_t pi = 1, pv = 0;
        for (idx_t r = 0; r < nr; ++r) {
            GidRow gr;
            gr.gid = bi[pi++];
            gr.nnz = bi[pi++];
            gr.cols.assign(&bi[pi], &bi[pi + gr.nnz]);
            pi += gr.nnz;
            gr.vals.assign(&bv[pv], &bv[pv + gr.nnz]);
            pv += gr.nnz;
            all_rows.push_back(std::move(gr));
        }
    }

    // Sort by global row index to maintain contiguous ordering
    std::sort(all_rows.begin(), all_rows.end(),
              [](const GidRow& a, const GidRow& b) { return a.gid < b.gid; });

    // Build CSR
    idx_t final_nlocal = static_cast<idx_t>(all_rows.size());
    std::vector<idx_t> new_rowptr(final_nlocal + 1);
    new_rowptr[0] = 0;
    int64_t final_nnz = 0;
    for (idx_t i = 0; i < final_nlocal; ++i) {
        final_nnz += all_rows[i].nnz;
        new_rowptr[i + 1] = final_nnz;
    }

    std::vector<idx_t> new_colidx(final_nnz);
    std::vector<val_t> new_val(final_nnz);
    int64_t pos = 0;
    for (idx_t i = 0; i < final_nlocal; ++i) {
        std::copy(all_rows[i].cols.begin(), all_rows[i].cols.end(), &new_colidx[pos]);
        std::copy(all_rows[i].vals.begin(), all_rows[i].vals.end(), &new_val[pos]);
        pos += all_rows[i].nnz;
    }

    // ── Step 8: Update outputs ──
    rowptr = std::move(new_rowptr);
    colidx = std::move(new_colidx);
    val = std::move(new_val);
    nlocal = final_nlocal;
    r_start = new_r_start;
    r_end = new_r_end;

    // Reset left/right for the new row range
    left = r_start;
    right = std::min(r_end, ncols);

    if (rank == 0) {
        std::printf("Redistribute: rank %d now has %d rows [%d,%d), "
                    "nnz=%lld\n", rank, nlocal, r_start, r_end, final_nnz);
    }
}

}  // namespace distspmv
