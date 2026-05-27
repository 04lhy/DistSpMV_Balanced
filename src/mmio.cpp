#include "mmio.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace distspmv {

namespace {

// Quick hash for coordinate deduplication
struct pair_hash {
    std::size_t operator()(const std::pair<idx_t, idx_t>& p) const {
        return static_cast<std::size_t>(p.first) * 31 +
               static_cast<std::size_t>(p.second);
    }
};

/// Trim leading whitespace from a C string.
const char* ltrim(const char* s) {
    while (*s && std::isspace(static_cast<unsigned char>(*s))) ++s;
    return s;
}

}  // namespace

CSRMatrix read_mtx(const std::string& filepath) {
    std::string path_copy = filepath;
    FILE* fp = std::fopen(path_copy.c_str(), "r");
    if (!fp) {
        std::fprintf(stderr, "[ERROR] Cannot open matrix file: %s\n",
                     filepath.c_str());
        std::exit(1);
    }

    char line[4096];
    bool is_symmetric = false;
    bool is_coordinate = true;
    idx_t nrows = 0, ncols = 0;
    int64_t nnz_header = 0;

    // ── Parse header ──
    while (std::fgets(line, sizeof(line), fp)) {
        const char* p = ltrim(line);
        if (p[0] == '\0') continue;

        if (p[0] == '%') {
            // Check for MatrixMarket banner
            if (std::strncmp(p, "%%MatrixMarket", 14) == 0) {
                // Format: %%MatrixMarket matrix coordinate real general
                if (std::strstr(p, "symmetric") ||
                    std::strstr(p, "skew-symmetric") ||
                    std::strstr(p, "hermitian")) {
                    is_symmetric = true;
                }
                if (std::strstr(p, "array")) {
                    is_coordinate = false;
                }
            }
            continue;
        }

        // First non-comment line: dimensions
        if (std::sscanf(p, "%d %d %lld", &nrows, &ncols, &nnz_header) >= 2) {
            break;
        }
    }

    if (nrows <= 0 || ncols <= 0) {
        std::fprintf(stderr, "[ERROR] Invalid matrix dimensions in %s\n",
                     filepath.c_str());
        std::fclose(fp);
        std::exit(1);
    }

    if (!is_coordinate) {
        // Array format: read nrows*ncols values, create dense->CSR
        std::vector<idx_t> rowptr(nrows + 1);
        std::vector<idx_t> colidx;
        std::vector<val_t> val;
        colidx.reserve(static_cast<std::size_t>(nrows) * ncols / 4);  // guess sparse
        val.reserve(colidx.capacity());

        int64_t idx = 0;
        double v;
        for (idx_t i = 0; i < nrows; ++i) {
            rowptr[i] = static_cast<idx_t>(colidx.size());
            for (idx_t j = 0; j < ncols; ++j) {
                if (std::fscanf(fp, "%lf", &v) != 1) {
                    std::fprintf(stderr, "[ERROR] Premature EOF in array data\n");
                    std::fclose(fp);
                    std::exit(1);
                }
                if (v != 0.0) {
                    colidx.push_back(j);
                    val.push_back(v);
                }
            }
        }
        rowptr[nrows] = static_cast<idx_t>(colidx.size());
        std::fclose(fp);

        return CSRMatrix{
            std::move(rowptr),
            std::move(colidx),
            std::move(val),
            nrows, ncols,
            static_cast<int64_t>(rowptr.back()),
        };
    }

    // ── Coordinate format ──
    // First pass: count entries per row, build rowptr
    std::vector<idx_t> rowptr(nrows + 1, 0);

    // Read all triples
    struct Triple {
        idx_t i, j;
        val_t v;
    };
    std::vector<Triple> triples;
    triples.reserve(static_cast<std::size_t>(nnz_header > 0 ? nnz_header * 2 : 1000000));

    idx_t ri, ci;
    double rv;
    while (std::fscanf(fp, "%d %d %lf", &ri, &ci, &rv) == 3) {
        ri -= 1;  // 1-based → 0-based
        ci -= 1;
        triples.push_back({ri, ci, rv});
        rowptr[ri + 1]++;
        if (is_symmetric && ri != ci) {
            triples.push_back({ci, ri, rv});
            rowptr[ci + 1]++;
        }
    }
    std::fclose(fp);

    // Build prefix-sum rowptr
    for (idx_t i = 0; i < nrows; ++i) {
        rowptr[i + 1] += rowptr[i];
    }

    // ── Second pass: fill colidx/val ──
    std::vector<idx_t> colidx(rowptr.back());
    std::vector<val_t> values(rowptr.back());
    std::vector<idx_t> cursor(rowptr.begin(), rowptr.begin() + nrows);

    for (const auto& t : triples) {
        idx_t pos = cursor[t.i]++;
        colidx[pos] = t.j;
        values[pos] = t.v;
    }

    // ── Sort columns within each row and deduplicate ──
    for (idx_t i = 0; i < nrows; ++i) {
        idx_t start = rowptr[i];
        idx_t end = rowptr[i + 1];
        if (end - start <= 1) continue;

        // Sort by column index
        std::vector<std::pair<idx_t, val_t>> row_entries;
        row_entries.reserve(end - start);
        for (idx_t j = start; j < end; ++j) {
            row_entries.emplace_back(colidx[j], values[j]);
        }
        std::sort(row_entries.begin(), row_entries.end());

        // Deduplicate: sum values for same column
        idx_t w = start;
        colidx[w] = row_entries[0].first;
        values[w] = row_entries[0].second;
        for (std::size_t k = 1; k < row_entries.size(); ++k) {
            if (row_entries[k].first == colidx[w]) {
                values[w] += row_entries[k].second;
            } else {
                ++w;
                colidx[w] = row_entries[k].first;
                values[w] = row_entries[k].second;
            }
        }
        rowptr[i] = start;
        // Compact
        idx_t new_end = w + 1;
        if (new_end < end) {
            colidx.erase(colidx.begin() + new_end, colidx.begin() + end);
            values.erase(values.begin() + new_end, values.begin() + end);
            // Adjust subsequent row pointers
            idx_t delta = end - new_end;
            for (idx_t k = i + 1; k <= nrows; ++k) {
                rowptr[k] -= delta;
            }
        }
    }

    // Remove zero entries
    for (idx_t i = 0; i < nrows; ++i) {
        idx_t& start = rowptr[i];
        idx_t end = rowptr[i + 1];
        idx_t w = start;
        for (idx_t j = start; j < end; ++j) {
            if (values[j] != 0.0) {
                if (w != j) {
                    colidx[w] = colidx[j];
                    values[w] = values[j];
                }
                ++w;
            }
        }
        idx_t removed = end - w;
        if (removed > 0) {
            colidx.erase(colidx.begin() + w, colidx.begin() + end);
            values.erase(values.begin() + w, values.begin() + end);
            for (idx_t k = i + 1; k <= nrows; ++k) {
                rowptr[k] -= removed;
            }
        }
    }

    int64_t total_nnz = rowptr.back();
    return CSRMatrix{
        std::move(rowptr),
        std::move(colidx),
        std::move(values),
        nrows, ncols, total_nnz,
    };
}

}  // namespace distspmv
