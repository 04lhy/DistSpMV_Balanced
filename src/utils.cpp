#include "utils.hpp"

#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <ctime>
#include <random>

namespace distspmv {

// ── Error computation ────────────────────────────────────────────────

double compute_error(const double* y_test, const double* y_ref, idx_t n) {
    double sq_err = 0.0, sq_ref = 0.0;
#pragma omp parallel for reduction(+ : sq_err, sq_ref) if (n > 10000)
    for (idx_t i = 0; i < n; ++i) {
        double diff = y_test[i] - y_ref[i];
        sq_err += diff * diff;
        sq_ref += y_ref[i] * y_ref[i];
    }
    double denom = std::sqrt(sq_ref);
    if (denom == 0.0) return std::sqrt(sq_err);
    return std::sqrt(sq_err) / denom;
}

// ── Serial SpMV (for verification) ───────────────────────────────────

void serial_spmv(const CSRMatrix& A, const double* x, double* y) {
    const idx_t* rowptr = A.rowptr.data();
    const idx_t* colidx = A.colidx.data();
    const val_t* val = A.val.data();
    idx_t nrows = A.nrows;

#pragma omp parallel for if (nrows > 10000)
    for (idx_t i = 0; i < nrows; ++i) {
        double acc = 0.0;
        for (idx_t j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            acc += val[j] * x[colidx[j]];
        }
        y[i] = acc;
    }
}

// ── Logging ──────────────────────────────────────────────────────────

void log_info(int rank, const char* fmt, ...) {
    // Timestamp
    std::time_t now = std::time(nullptr);
    char time_buf[16];
    std::strftime(time_buf, sizeof(time_buf), "%H:%M:%S",
                  std::localtime(&now));

    std::fprintf(stdout, "[rank %2d] %s INFO  ", rank, time_buf);

    std::va_list args;
    va_start(args, fmt);
    std::vfprintf(stdout, fmt, args);
    va_end(args);
    std::fprintf(stdout, "\n");
    std::fflush(stdout);
}

void set_seed(int seed) {
    std::srand(static_cast<unsigned>(seed));
}

}  // namespace distspmv
