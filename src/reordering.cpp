#include "reordering.hpp"

#include <algorithm>
#include <cstdio>
#include <deque>
#include <numeric>
#include <vector>

#ifdef DISTSPMV_HAS_METIS
#include <metis.h>
#endif

namespace distspmv {

namespace {

/// RCM permutation using the CSR structure directly as adjacency.
/// Memory-efficient: O(n + max_degree) instead of building a separate
/// adjacency matrix.  Assumes the matrix is structurally symmetric
/// (A[i,j] != 0 iff A[j,i] != 0), which holds for SuiteSparse matrices.
std::vector<idx_t> compute_rcm_permutation(const CSRMatrix& A) {
    idx_t n = A.nrows;
    std::fprintf(stdout, "RCM: n=%d nnz=%lld\n", n, (long long)A.nnz);
    std::fflush(stdout);

    // ── Count degree per vertex (off-diagonal neighbours only) ──
    std::fprintf(stdout, "RCM: counting degrees...\n"); std::fflush(stdout);
    std::vector<int> degree(n, 0);
    for (idx_t i = 0; i < n; ++i) {
        int d = 0;
        for (idx_t j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {
            if (A.colidx[j] != i) d++;
        }
        degree[i] = d;
    }
    std::fprintf(stdout, "RCM: degrees done.\n"); std::fflush(stdout);

    // ── RCM BFS traversal ──
    std::fprintf(stdout, "RCM: BFS traversal...\n"); std::fflush(stdout);
    std::vector<idx_t> order;
    order.reserve(n);
    std::vector<bool> visited(n, false);
    std::vector<std::pair<int, idx_t>> nb;  // reused per vertex

    auto process_component = [&](idx_t root) {
        std::deque<idx_t> q;
        q.push_back(root);
        visited[root] = true;

        while (!q.empty()) {
            idx_t v = q.front();
            q.pop_front();
            order.push_back(v);

            // Collect unvisited neighbours sorted by increasing degree
            nb.clear();
            for (idx_t j = A.rowptr[v]; j < A.rowptr[v + 1]; ++j) {
                idx_t w = A.colidx[j];
                if (w != v && !visited[w]) {
                    nb.emplace_back(degree[w], w);
                }
            }
            std::sort(nb.begin(), nb.end());

            for (auto& [_, w] : nb) {
                visited[w] = true;
                q.push_back(w);
            }
        }
    };

    // Find starting vertex with minimum degree
    idx_t root = 0;
    int min_deg = degree[0];
    for (idx_t i = 1; i < n; ++i) {
        if (degree[i] < min_deg) {
            min_deg = degree[i];
            root = i;
        }
    }

    process_component(root);

    // Handle any disconnected components
    for (idx_t i = 0; i < n; ++i) {
        if (!visited[i]) process_component(i);
    }

    // ── Reverse for RCM ──
    std::reverse(order.begin(), order.end());

    std::vector<idx_t> perm(n);
    for (idx_t i = 0; i < n; ++i) {
        perm[order[i]] = i;
    }
    return perm;
}

#ifdef DISTSPMV_HAS_METIS
// Build undirected adjacency (A + A^T pattern, self-loops removed)
// Returns vector-of-vectors: adj[i] = list of neighboring column indices
std::vector<std::vector<idx_t>> build_adjacency(const CSRMatrix& A) {
    idx_t n = A.nrows;
    std::vector<std::vector<idx_t>> adj(n);
    for (idx_t i = 0; i < n; ++i) {
        for (idx_t p = A.rowptr[i]; p < A.rowptr[i + 1]; ++p) {
            idx_t j = A.colidx[p];
            if (j != i) {  // skip self-loops
                adj[i].push_back(j);
                adj[j].push_back(i);  // symmetrize: A + A^T
            }
        }
    }
    // Deduplicate each row
    for (idx_t i = 0; i < n; ++i) {
        std::sort(adj[i].begin(), adj[i].end());
        adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
    }
    return adj;
}
#endif

}  // namespace

CSRMatrix reorder_matrix(const CSRMatrix& A, const std::string& method,
                         int nparts, int seed) {
    idx_t n = A.nrows;

    if (method == "none") {
        std::fprintf(stdout, "Reordering: none (identity permutation)\n");
        return CSRMatrix{
            A.rowptr, A.colidx, A.val,
            A.nrows, A.ncols, A.nnz,
        };
    }

    std::vector<idx_t> perm;

    if (method == "rcm") {
        std::fprintf(stdout, "Reordering: RCM ...\n");
        std::fflush(stdout);
        perm = compute_rcm_permutation(A);
        std::fprintf(stdout, "Reordering: RCM done, applying permutation ...\n");
        std::fflush(stdout);
    }
#ifdef DISTSPMV_HAS_METIS
    else if (method == "metis") {
        if (nparts <= 1) {
            std::fprintf(stdout, "Reordering: metis skipped (nparts=%d <= 1)\n", nparts);
            return CSRMatrix{
                A.rowptr, A.colidx, A.val,
                A.nrows, A.ncols, A.nnz,
            };
        }
        std::fprintf(stdout, "Reordering: METIS (nparts=%d)...\n", nparts);
        idx_t nvtxs = n;
        idx_t ncon = 1;

        // Build adjacency in CSR for METIS
        auto adj = build_adjacency(A);
        std::vector<idx_t> xadj(n + 1);
        std::vector<idx_t> adjncy;
        for (idx_t i = 0; i < n; ++i) {
            xadj[i] = static_cast<idx_t>(adjncy.size());
            adjncy.insert(adjncy.end(), adj[i].begin(), adj[i].end());
        }
        xadj[n] = static_cast<idx_t>(adjncy.size());

        std::vector<idx_t> parts(n);
        int objval;
        int result = METIS_PartGraphKway(
            &nvtxs, &ncon, xadj.data(), adjncy.data(),
            nullptr, nullptr, nullptr, &nparts,
            nullptr, nullptr, nullptr, &objval, parts.data());

        if (result != METIS_OK) {
            std::fprintf(stderr, "[ERROR] METIS partitioning failed (err=%d)\n", result);
            std::exit(1);
        }

        // Order by (partition_id, original_index)
        std::vector<idx_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](idx_t a, idx_t b) {
                      if (parts[a] != parts[b]) return parts[a] < parts[b];
                      return a < b;
                  });

        perm.resize(n);
        for (idx_t i = 0; i < n; ++i) {
            perm[order[i]] = i;
        }
    }
#endif
    else {
        std::fprintf(stderr, "[ERROR] Unknown reordering method: %s\n",
                     method.c_str());
        std::exit(1);
    }

    // Apply symmetric permutation: A_reordered = P @ A @ P^T
    std::fprintf(stdout, "Reordering: counting nnz per new row...\n");
    std::fflush(stdout);

    // First pass: count nnz per row in reordered matrix
    std::vector<idx_t> new_rowptr(n + 1, 0);
    for (idx_t i = 0; i < n; ++i) {
        idx_t new_i = perm[i];
        idx_t js = A.rowptr[i];
        idx_t je = A.rowptr[i + 1];
        for (idx_t j = js; j < je; ++j) {
            idx_t c = A.colidx[j];
            if (c < 0 || c >= n) {
                std::fprintf(stderr, "ERROR: colidx[%d]=%d out of [0,%d)\n",
                            j, c, n);
                std::exit(1);
            }
            idx_t new_c = perm[c];
            new_rowptr[new_i + 1]++;
        }
    }

    std::fprintf(stdout, "Reordering: prefix sum...\n"); std::fflush(stdout);
    for (idx_t i = 0; i < n; ++i) {
        new_rowptr[i + 1] += new_rowptr[i];
    }

    std::fprintf(stdout, "Reordering: filling colidx/val (total_nnz=%d)...\n",
                new_rowptr.back());
    std::fflush(stdout);

    // Second pass: fill colidx/val
    std::vector<idx_t> new_colidx(new_rowptr.back());
    std::vector<val_t> new_val(new_rowptr.back());
    std::vector<idx_t> cursor(new_rowptr.begin(), new_rowptr.begin() + n);

    for (idx_t i = 0; i < n; ++i) {
        idx_t new_i = perm[i];
        for (idx_t j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {
            idx_t new_c = perm[A.colidx[j]];
            idx_t pos = cursor[new_i]++;
            new_colidx[pos] = new_c;
            new_val[pos] = A.val[j];
        }
    }

    // Sort columns within each row
    for (idx_t i = 0; i < n; ++i) {
        idx_t start = new_rowptr[i];
        idx_t end = new_rowptr[i + 1];
        if (end - start <= 1) continue;

        std::vector<std::pair<idx_t, val_t>> entries;
        entries.reserve(end - start);
        for (idx_t j = start; j < end; ++j) {
            entries.emplace_back(new_colidx[j], new_val[j]);
        }
        std::sort(entries.begin(), entries.end());
        for (idx_t j = start; j < end; ++j) {
            new_colidx[j] = entries[j - start].first;
            new_val[j] = entries[j - start].second;
        }
    }
    auto total_nnz = static_cast<int64_t>(new_rowptr.back());
    return CSRMatrix{
        std::move(new_rowptr),
        std::move(new_colidx),
        std::move(new_val),
        n, n,
        total_nnz,
    };
}

}  // namespace distspmv
