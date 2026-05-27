#include "spmv_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace distspmv {

// ── Constructor ──────────────────────────────────────────────────────

DistSpMVSolver::DistSpMVSolver(
    const idx_t* rowptr, const idx_t* colidx, const val_t* val,
    idx_t nlocal, idx_t r_start, idx_t r_end,
    idx_t left, idx_t right,
    const CommSchedule& sched,
    MPI_Comm comm, int n_threads)
    : nlocal_(nlocal), r_start_(r_start), r_end_(r_end),
      left_(left), right_(right),
      sched_(sched), comm_(comm), n_threads_(n_threads) {

    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &nprocs_);

    nnz_local_ = (rowptr != nullptr) ? rowptr[nlocal] : 0;
    diag_len_ = right - left;

    // Copy CSR data
    rowptr_.assign(rowptr, rowptr + nlocal + 1);
    colidx_.assign(colidx, colidx + nnz_local_);
    val_.assign(val, val + nnz_local_);

    // ── Gather all ranks' initial row ranges for column ownership ──
    idx_t my_bounds[2] = {r_start_, r_end_};
    std::vector<idx_t> all_bounds(static_cast<std::size_t>(nprocs_) * 2);
    MPI_Allgather(my_bounds, 2, MPI_INT, all_bounds.data(), 2,
                  MPI_INT, comm_);

    owner_r_start_.resize(nprocs_);
    owner_r_end_.resize(nprocs_);
    for (int q = 0; q < nprocs_; ++q) {
        owner_r_start_[q] = all_bounds[q * 2];
        owner_r_end_[q] = all_bounds[q * 2 + 1];
    }

    // ── Build unified x_buf layout ──
    total_remote_ = 0;
    for (auto& [q, cols] : sched_.recvid) {
        remote_offset_[q] = total_remote_;
        total_remote_ += static_cast<idx_t>(cols.size());
    }

    idx_t x_buf_len = diag_len_ + total_remote_;
    x_buf_.resize(x_buf_len, 0.0);

    // ── Build local_pos array (Algorithm 4 preparation) ──
    local_pos_.resize(nnz_local_);
    for (int64_t j = 0; j < nnz_local_; ++j) {
        idx_t c = colidx_[j];
        if (left_ <= c && c < right_) {
            local_pos_[j] = c - left_;
        } else {
            int owner = find_owner_fast(c);
            if (owner == rank_) {
                local_pos_[j] = c - left_;
            } else {
                auto it_peer = sched_.recv_idx_map.find(owner);
                if (it_peer != sched_.recv_idx_map.end()) {
                    auto it_col = it_peer->second.find(c);
                    if (it_col != it_peer->second.end()) {
                        local_pos_[j] = diag_len_ + remote_offset_.at(owner)
                                        + it_col->second;
                    } else {
                        local_pos_[j] = 0;  // shouldn't happen
                    }
                } else {
                    local_pos_[j] = 0;
                }
            }
        }
    }

    // ── Build thread partitioning by nnz per row ──
    thread_bounds_.push_back(0);
    if (n_threads_ > 1) {
        int64_t total_nnz = nnz_local_;
        int64_t target_per_thread = std::max(int64_t(1),
                                              total_nnz / n_threads_);
        int64_t acc = 0;
        for (idx_t i = 0; i < nlocal_; ++i) {
            acc += rowptr_[i + 1] - rowptr_[i];
            if (acc >= target_per_thread &&
                static_cast<int>(thread_bounds_.size()) < n_threads_) {
                thread_bounds_.push_back(i + 1);
                acc = 0;
            }
        }
    }
    thread_bounds_.push_back(nlocal_);

    std::printf("[rank %2d] DistSpMVSolver: nlocal=%d nnz_local=%lld "
                "diag_len=%d total_remote=%d n_threads=%d\n",
                rank_, nlocal_, nnz_local_, diag_len_,
                total_remote_, n_threads_);
}

// ── Public API ───────────────────────────────────────────────────────

void DistSpMVSolver::multiply(const double* x_global, double* y_out) {
    exchange_remote(x_global);
    local_spmv(y_out);
}

std::pair<double, double> DistSpMVSolver::multiply_benchmark(
    const double* x_global, int n_warmup, int n_repeat) {

    // Warm-up
    std::vector<double> y_tmp(nlocal_);
    for (int i = 0; i < n_warmup; ++i) {
        multiply(x_global, y_tmp.data());
    }

    // Timed runs
    std::vector<double> times(n_repeat);
    for (int i = 0; i < n_repeat; ++i) {
        double t0 = MPI_Wtime();
        multiply(x_global, y_tmp.data());
        times[i] = MPI_Wtime() - t0;
    }

    double mean = 0.0, m2 = 0.0;
    for (int i = 0; i < n_repeat; ++i) {
        double delta = times[i] - mean;
        mean += delta / (i + 1);
        double delta2 = times[i] - mean;
        m2 += delta * delta2;
    }
    double std_dev = (n_repeat > 1) ? std::sqrt(m2 / (n_repeat - 1)) : 0.0;

    return {mean, std_dev};
}

// ── Algorithm 3: MPI Communication ──────────────────────────────────

void DistSpMVSolver::exchange_remote(const double* x_global) {
    // 1. Copy local x elements into x_buf
    for (idx_t c = left_; c < right_; ++c) {
        x_buf_[c - left_] = x_global[c];
    }

    // 2. Pack send buffers
    std::unordered_map<int, std::vector<double>> send_bufs;
    for (auto& [q, cols] : sched_.sendid) {
        std::size_t n = cols.size();
        if (n == 0) continue;
        send_bufs[q].resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            send_bufs[q][k] = x_global[cols[k]];
        }
    }

    // 3. Post receives for incoming remote x elements
    std::unordered_map<int, std::vector<double>> recv_bufs;
    std::vector<MPI_Request> recv_reqs;

    for (auto& [q, cols] : sched_.recvid) {
        std::size_t n = cols.size();
        if (n == 0) continue;
        recv_bufs[q].resize(n);
        MPI_Request req;
        MPI_Irecv(recv_bufs[q].data(), static_cast<int>(n), MPI_DOUBLE,
                  q, 200 + q, comm_, &req);
        recv_reqs.push_back(req);
    }

    // 4. Post sends
    std::vector<MPI_Request> send_reqs;
    for (auto& [q, buf] : send_bufs) {
        MPI_Request req;
        MPI_Isend(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE,
                  q, 200 + rank_, comm_, &req);
        send_reqs.push_back(req);
    }

    // 5. Wait for all to complete
    std::vector<MPI_Request> all_reqs;
    all_reqs.reserve(recv_reqs.size() + send_reqs.size());
    all_reqs.insert(all_reqs.end(), recv_reqs.begin(), recv_reqs.end());
    all_reqs.insert(all_reqs.end(), send_reqs.begin(), send_reqs.end());

    MPI_Waitall(static_cast<int>(all_reqs.size()), all_reqs.data(),
                MPI_STATUSES_IGNORE);

    // 6. Unpack received data into x_buf
    for (auto& [q, buf] : recv_bufs) {
        idx_t offset = diag_len_ + remote_offset_.at(q);
        for (std::size_t k = 0; k < buf.size(); ++k) {
            x_buf_[offset + k] = buf[k];
        }
    }
}

// ── Algorithm 4: Multi-threaded Local SpMV ──────────────────────────

void DistSpMVSolver::local_spmv(double* y_out) {
    const int n_threads = n_threads_;
    const idx_t nlocal = nlocal_;
    const idx_t* rowptr = rowptr_.data();
    const idx_t* colidx = colidx_.data();
    const val_t* val = val_.data();
    const idx_t* local_pos = local_pos_.data();
    const double* x_buf = x_buf_.data();

    // Zero output
    std::fill(y_out, y_out + nlocal, 0.0);

    int n_chunks = static_cast<int>(thread_bounds_.size()) - 1;

#pragma omp parallel num_threads(n_threads) if(n_threads > 1)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        if (tid >= n_chunks) {
            // extra threads idle — skip
        } else {
            idx_t start_row = thread_bounds_[tid];
            idx_t end_row = thread_bounds_[tid + 1];

            // Private accumulator for this thread's rows
            for (idx_t i = start_row; i < end_row; ++i) {
                double acc = 0.0;
                idx_t js = rowptr[i];
                idx_t je = rowptr[i + 1];
                for (idx_t j = js; j < je; ++j) {
                    acc += val[j] * x_buf[local_pos[j]];
                }
                y_out[i] += acc;
            }
        }
    }
}

// ── Accessors / Helpers ──────────────────────────────────────────────

idx_t DistSpMVSolver::nnz_diag() const {
    idx_t count = 0;
    for (int64_t j = 0; j < nnz_local_; ++j) {
        idx_t c = colidx_[j];
        if (left_ <= c && c < right_) count++;
    }
    return count;
}

int DistSpMVSolver::find_owner_fast(idx_t col) const {
    for (int q = 0; q < nprocs_; ++q) {
        if (owner_r_start_[q] <= col && col < owner_r_end_[q]) return q;
    }
    return rank_;
}

}  // namespace distspmv
