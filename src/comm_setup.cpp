#include "comm_setup.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <unordered_map>
#include <vector>

namespace distspmv {

// Helper: find owner rank for global column c (binary search on monotonic ranges)
static int find_owner(idx_t c, const idx_t* all_r_start,
                      const idx_t* all_r_end, int nprocs) {
    auto it = std::upper_bound(all_r_start, all_r_start + nprocs, c);
    if (it == all_r_start) return 0;
    int q = static_cast<int>(it - all_r_start) - 1;
    return (c < all_r_end[q]) ? q : nprocs - 1;
}

CommSchedule build_schedule(
    const idx_t* rowptr, const idx_t* colidx,
    idx_t nlocal, idx_t r_start, idx_t r_end,
    idx_t left, idx_t right, MPI_Comm comm) {

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    // ── Step 1: Allgather boundaries & initial row ranges ──
    idx_t my_info[4] = {left, right, r_start, r_end};
    std::vector<idx_t> all_info(static_cast<std::size_t>(nprocs) * 4);
    MPI_Allgather(my_info, 4, MPI_INT, all_info.data(), 4, MPI_INT,
                  comm);

    std::vector<idx_t> all_left(nprocs), all_right(nprocs);
    std::vector<idx_t> all_r_start(nprocs), all_r_end(nprocs);
    for (int q = 0; q < nprocs; ++q) {
        all_left[q]   = all_info[q * 4 + 0];
        all_right[q]  = all_info[q * 4 + 1];
        all_r_start[q] = all_info[q * 4 + 2];
        all_r_end[q]  = all_info[q * 4 + 3];
    }

    // ── Step 2: Build recvid sets (columns needed from each remote rank) ──
    std::unordered_map<int, std::set<idx_t>> recv_sets;
    for (int q = 0; q < nprocs; ++q) {
        if (q != rank) recv_sets[q] = std::set<idx_t>();
    }

    for (idx_t i = 0; i < nlocal; ++i) {
        for (idx_t j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            idx_t c = colidx[j];
            if (left <= c && c < right) continue;  // diagonal block

            int owner = find_owner(c, all_r_start.data(), all_r_end.data(),
                                   nprocs);
            if (owner == rank) continue;  // we own it

            recv_sets[owner].insert(c);
        }
    }

    // Build recvid arrays and recv_idx_map
    CommSchedule sched;

    for (auto& [q, cols] : recv_sets) {
        if (cols.empty()) continue;
        sched.recvid[q].assign(cols.begin(), cols.end());
        // recvid is already sorted (set is ordered)
        for (std::size_t idx = 0; idx < sched.recvid[q].size(); ++idx) {
            sched.recv_idx_map[q][sched.recvid[q][idx]] = static_cast<int>(idx);
        }
    }

    // ── Step 3: Alltoall exchange counts ──
    std::vector<int> send_counts(nprocs, 0);
    for (auto& [q, cols] : sched.recvid) {
        send_counts[q] = static_cast<int>(cols.size());
    }
    std::vector<int> recv_counts(nprocs, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                 comm);
    // recv_counts[q] = how many columns rank q needs from US (|sendid[q]|)

    // ── Step 4: Exchange column-index lists via non-blocking P2P ──
    std::vector<MPI_Request> requests;
    std::unordered_map<int, std::vector<idx_t>> recv_bufs;

    // Post receives for incoming column lists
    for (int q = 0; q < nprocs; ++q) {
        if (q == rank) continue;
        int cnt = recv_counts[q];
        if (cnt > 0) {
            recv_bufs[q].resize(cnt);
            MPI_Request req;
            MPI_Irecv(recv_bufs[q].data(), cnt, MPI_INT, q, 100 + q,
                      comm, &req);
            requests.push_back(req);
        }
    }

    // Post sends of our recvid lists
    for (auto& [q, cols] : sched.recvid) {
        MPI_Request req;
        MPI_Isend(cols.data(), static_cast<int>(cols.size()), MPI_INT,
                  q, 100 + rank, comm, &req);
        requests.push_back(req);
    }

    // Wait for all exchanges
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                MPI_STATUSES_IGNORE);

    // Build sendid from received buffers
    for (auto& [q, buf] : recv_bufs) {
        if (!buf.empty()) {
            sched.sendid[q] = std::move(buf);
        }
    }

    // ── Statistics ──
    int64_t total_recv = 0, total_send = 0;
    for (auto& [_, cols] : sched.recvid) total_recv += cols.size();
    for (auto& [_, cols] : sched.sendid) total_send += cols.size();
    if (rank == 0) {
        std::printf("[rank %2d] Algo2: send to %zu peers (%lld elems), "
                    "recv from %zu peers (%lld elems)\n",
                    rank, sched.sendid.size(), total_send,
                    sched.recvid.size(), total_recv);
    }

    return sched;
}

}  // namespace distspmv
