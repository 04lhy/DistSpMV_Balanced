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

    // Field types: 0=real, 1=pattern, 2=integer, 3=complex
    enum { FIELD_REAL, FIELD_PATTERN, FIELD_INTEGER, FIELD_COMPLEX } field = FIELD_REAL;

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
                if (std::strstr(p, "pattern")) {
                    field = FIELD_PATTERN;
                } else if (std::strstr(p, "integer")) {
                    field = FIELD_INTEGER;
                } else if (std::strstr(p, "complex")) {
                    field = FIELD_COMPLEX;
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
    while (true) {
        int nread = 0;
        if (field == FIELD_PATTERN) {
            nread = std::fscanf(fp, "%d %d", &ri, &ci);
            rv = 1.0;
        } else if (field == FIELD_INTEGER) {
            int iv;
            nread = std::fscanf(fp, "%d %d %d", &ri, &ci, &iv);
            rv = static_cast<double>(iv);
        } else if (field == FIELD_COMPLEX) {
            double imag;
            nread = std::fscanf(fp, "%d %d %lf %lf", &ri, &ci, &rv, &imag);
            // Take real part only
        } else {
            nread = std::fscanf(fp, "%d %d %lf", &ri, &ci, &rv);
        }
        if (nread < 2) break;
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
    // Build fresh arrays to avoid O(nrows × nnz) from repeated erase.
    std::vector<idx_t> new_rowptr(nrows + 1);
    std::vector<idx_t> new_colidx;
    std::vector<val_t> new_values;
    new_colidx.reserve(rowptr.back());
    new_values.reserve(rowptr.back());

    for (idx_t i = 0; i < nrows; ++i) {
        new_rowptr[i] = static_cast<idx_t>(new_colidx.size());
        idx_t start = rowptr[i];
        idx_t end = rowptr[i + 1];
        if (end == start) continue;

        // Sort by column index
        std::vector<std::pair<idx_t, val_t>> row_entries;
        row_entries.reserve(end - start);
        for (idx_t j = start; j < end; ++j) {
            row_entries.emplace_back(colidx[j], values[j]);
        }
        std::sort(row_entries.begin(), row_entries.end());

        // Deduplicate: sum values for same column, skip zeros
        val_t accum = row_entries[0].second;
        idx_t cur_col = row_entries[0].first;
        for (std::size_t k = 1; k < row_entries.size(); ++k) {
            if (row_entries[k].first == cur_col) {
                accum += row_entries[k].second;
            } else {
                if (accum != 0.0) {
                    new_colidx.push_back(cur_col);
                    new_values.push_back(accum);
                }
                cur_col = row_entries[k].first;
                accum = row_entries[k].second;
            }
        }
        if (accum != 0.0) {
            new_colidx.push_back(cur_col);
            new_values.push_back(accum);
        }
    }
    new_rowptr[nrows] = static_cast<idx_t>(new_colidx.size());

    int64_t total_nnz = new_rowptr.back();
    return CSRMatrix{
        std::move(new_rowptr),
        std::move(new_colidx),
        std::move(new_values),
        nrows, ncols, total_nnz,
    };
}

}  // namespace distspmv
