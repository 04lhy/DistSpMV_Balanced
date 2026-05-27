#pragma once
/**
 * Minimal Matrix Market (.mtx) reader.
 *
 * Supports "coordinate" and "array" formats, "real" field, "general" /
 * "symmetric" symmetry.  1-based MTX indices are converted to 0-based.
 */

#include <string>

#include "types.hpp"

namespace distspmv {

/// Read a Matrix Market file and return a CSRMatrix.  Symmetric matrices
/// are expanded to general form by duplicating off-diagonal entries.
CSRMatrix read_mtx(const std::string& filepath);

}  // namespace distspmv
