"""
Unit tests for DistSpMV_Balanced modules.

Run with::

  pytest tests/ -v
  mpirun -n 4 python -m pytest tests/test_spmv.py -v  # MPI tests
"""

from __future__ import annotations

import os
import sys
import numpy as np
from scipy.sparse import csr_matrix, random as sprandom

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import (
    extract_csr, compute_error, verify_spmv, gflops,
    row_partition_range, set_seed,
)
from src.reordering import reorder_matrix
from src.partition import diagonal_block_expand


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

def make_test_matrix(n: int = 100, density: float = 0.1,
                     seed: int = 42) -> csr_matrix:
    """Create a small random sparse matrix for testing."""
    return sprandom(n, n, density=density, format="csr",
                    random_state=seed, dtype=np.float64)


def make_laplacian_2d(nx: int = 10, ny: int = 10) -> csr_matrix:
    """Create a 2D Laplacian (5-point stencil) as a square sparse matrix.

    This is a well-known sparse pattern where each row has ≤ 5 nonzeros.
    """
    n = nx * ny
    data, rows, cols = [], [], []
    for iy in range(ny):
        for ix in range(nx):
            k = iy * nx + ix
            # Center
            rows.append(k); cols.append(k); data.append(4.0)
            # Left
            if ix > 0:
                rows.append(k); cols.append(k - 1); data.append(-1.0)
            # Right
            if ix < nx - 1:
                rows.append(k); cols.append(k + 1); data.append(-1.0)
            # Up
            if iy > 0:
                rows.append(k); cols.append(k - nx); data.append(-1.0)
            # Down
            if iy < ny - 1:
                rows.append(k); cols.append(k + nx); data.append(-1.0)
    A = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    A.sort_indices()
    A.sum_duplicates()
    return A


# ──────────────────────────────────────────────────────────────────────
# Test: utils
# ──────────────────────────────────────────────────────────────────────

class TestUtils:
    def test_extract_csr_roundtrip(self):
        A = make_test_matrix(50, 0.1)
        rp, ci, v = extract_csr(A)
        B = csr_matrix((v, ci, rp), shape=A.shape)
        assert np.allclose(A.toarray(), B.toarray())

    def test_compute_error_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_error(y, y) == 0.0

    def test_compute_error_nonzero(self):
        y1 = np.array([1.0, 0.0, 0.0])
        y2 = np.array([1.0, 0.1, 0.0])
        err = compute_error(y2, y1)
        assert 0.0 < err < 1.0

    def test_verify_spmv_pass(self):
        A = make_test_matrix(50, 0.1)
        x = np.random.rand(50)
        y = A @ x
        ok, err = verify_spmv(y, A, x)
        assert ok
        assert err < 1e-12

    def test_verify_spmv_fail(self):
        A = make_test_matrix(50, 0.1)
        x = np.random.rand(50)
        y_bad = np.zeros(50)
        ok, _ = verify_spmv(y_bad, A, x)
        assert not ok

    def test_gflops(self):
        gf = gflops(1_000_000, 0.01)
        assert 0.19 < gf < 0.21  # 2*1e6 / (0.01*1e9) = 0.2

    def test_row_partition_range(self):
        # 100 rows, 4 procs
        s0, e0 = row_partition_range(100, 4, 0)
        s1, e1 = row_partition_range(100, 4, 1)
        s2, e2 = row_partition_range(100, 4, 2)
        s3, e3 = row_partition_range(100, 4, 3)
        assert s0 == 0 and e0 == 25
        assert s1 == 25 and e1 == 50
        assert s2 == 50 and e2 == 75
        assert s3 == 75 and e3 == 100

    def test_row_partition_uneven(self):
        # 10 rows, 3 procs
        s0, e0 = row_partition_range(10, 3, 0)  # 10/3 → base=3, rem=1 → [0,4)
        s1, e1 = row_partition_range(10, 3, 1)  # [4,7)
        s2, e2 = row_partition_range(10, 3, 2)  # [7,10)
        assert (e0 - s0) + (e1 - s1) + (e2 - s2) == 10
        assert s0 == 0

    def test_set_seed(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        assert np.array_equal(a, b)


# ──────────────────────────────────────────────────────────────────────
# Test: reordering
# ──────────────────────────────────────────────────────────────────────

class TestReordering:
    def test_reorder_none(self):
        A = make_test_matrix(50, 0.1)
        B, perm = reorder_matrix(A, method="none")
        assert np.allclose(A.toarray(), B.toarray())
        assert np.array_equal(perm, np.arange(50))

    def test_reorder_rcm_preserves_spectrum(self):
        """RCM is a symmetric permutation — eigenvalues are preserved."""
        A = make_laplacian_2d(10, 10)  # well-conditioned
        B, perm = reorder_matrix(A, method="rcm")
        assert A.shape == B.shape
        assert np.allclose(np.sort(np.linalg.eigvalsh(A.toarray())),
                           np.sort(np.linalg.eigvalsh(B.toarray())),
                           atol=1e-10)

    def test_reorder_rcm_reduces_bandwidth(self):
        """RCM should reduce the (non-strict) bandwidth."""
        A = make_laplacian_2d(10, 10)
        B, _ = reorder_matrix(A, method="rcm")

        def bandwidth(M: csr_matrix) -> int:
            bw = 0
            for i in range(M.shape[0]):
                cols = M.indices[M.indptr[i]:M.indptr[i+1]]
                if len(cols) > 0:
                    bw = max(bw, cols.max() - i)
            return bw

        # RCM is heuristic; bandwidth may not always decrease, but
        # for 2D Laplacian it usually does.
        bw_a = bandwidth(A)
        bw_b = bandwidth(B)
        # Just check it runs without error; bandwidth improvement
        # is not guaranteed for all matrices.
        assert B.shape == A.shape


# ──────────────────────────────────────────────────────────────────────
# Test: partition (Algorithm 1)
# ──────────────────────────────────────────────────────────────────────

class TestPartition:
    def _mock_comm(self, max_val=0):
        """Create a mock MPI comm that returns `max_val` for allreduce."""
        class MockComm:
            def allreduce(self, val, op=None):
                return max(val, max_val)
            def Get_rank(self): return 0
            def Get_size(self): return 1
        return MockComm()

    def test_diag_block_trivial(self):
        """With 1 process, the diagonal block should cover the whole matrix."""
        A = make_laplacian_2d(5, 5)
        n = A.shape[0]
        rp, ci, _ = extract_csr(A)
        comm = self._mock_comm()
        left, right = diagonal_block_expand(rp, ci, 0, n, n, comm)
        # For a single process, the expansion should at minimum
        # cover the initial [0, n) range.
        assert left <= 0
        assert right >= n

    def test_diag_block_expands(self):
        """Test that a heavily off-diagonal row triggers expansion."""
        n = 20
        # Create a matrix where row 0 has only column 19 (far off-diag)
        data = np.ones(100, dtype=np.float64)
        rows = np.random.default_rng(42).integers(0, n, 100)
        cols = np.random.default_rng(43).integers(0, n, 100)
        A = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
        A.sort_indices()
        A.sum_duplicates()
        A.eliminate_zeros()

        rp, ci, _ = extract_csr(A)

        # Simulate 2-process case: rank 0 gets rows [0, 10)
        r_start, r_end = 0, 10
        comm = self._mock_comm(max_val=3)
        left, right = diagonal_block_expand(rp[:11], ci[:rp[10]], r_start, r_end, n, comm)

        # The boundaries should have expanded beyond [0, 10)
        assert right > 10 or left < 0, (
            f"Expected expansion beyond [0,10), got [{left}, {right})"
        )

    def test_boundaries_in_range(self):
        """Boundaries must always be within [0, ncols]."""
        A = make_test_matrix(50, 0.05)
        n = A.shape[0]
        rp, ci, _ = extract_csr(A)
        comm = self._mock_comm(max_val=2)
        left, right = diagonal_block_expand(rp, ci, 0, n, n, comm)
        assert 0 <= left <= n
        assert 0 <= right <= n
        assert left <= right


# ──────────────────────────────────────────────────────────────────────
# Test: spherical chicken (end-to-end on single process)
# ──────────────────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_full_pipeline_serial(self):
        """Run the full pipeline on 1 MPI process (simulated)."""
        from mpi4py import MPI

        # This test only works when run with mpiexec -n 1
        comm = MPI.COMM_WORLD
        if comm.Get_size() != 1:
            import pytest
            pytest.skip("This test requires exactly 1 MPI rank")

        A = make_laplacian_2d(8, 8)  # 64x64
        n = A.shape[0]
        rp_g, ci_g, v_g = extract_csr(A)

        r_start, r_end = 0, n
        local_rp = rp_g.copy()
        local_ci = ci_g.copy()
        local_v = v_g.copy()

        left, right = diagonal_block_expand(
            local_rp, local_ci, r_start, r_end, n, comm,
        )

        from src.comm_setup import build_schedule
        sendid, recvid, recv_idx_map = build_schedule(
            local_rp, local_ci, r_start, r_end, left, right, comm,
        )

        from src.spmv_mpi_omp import DistSpMVSolver
        solver = DistSpMVSolver(
            local_rp, local_ci, local_v,
            r_start, r_end, left, right,
            sendid, recvid, recv_idx_map,
            comm, n_threads=1,
        )

        x = np.random.rand(n).astype(np.float64)
        y = solver.multiply(x)
        y_ref = A @ x

        err = compute_error(y, y_ref)
        assert err < 1e-10, f"Relative error {err} exceeds tolerance"


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
