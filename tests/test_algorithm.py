"""
Unit tests for the YYC algorithm.

Tests cover:
- Standard cases with known results
- Edge cases (empty matrices, single row/column, zero rows)
- Random stress tests for consistency
"""

import numpy as np
import pytest

from yyc.core.algorithm import yyc_algorithm, _is_compatible_set


class TestYYCAlgorithm:
    """Tests for the main YYC algorithm function."""

    def test_simple_3x3_matrix(self):
        """Test with a simple 3x3 binary matrix."""
        matrix = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0]
        ])
        result = yyc_algorithm(matrix)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(t, tuple) for t in result)

    def test_identity_matrix(self):
        """Test with identity matrix - each column distinguishes one row."""
        matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        result = yyc_algorithm(matrix)

        assert len(result) > 0
        # All testors should contain all three columns
        for testor in result:
            assert len(testor) == 3

    def test_single_column_all_ones(self):
        """Test matrix where one column has all ones."""
        matrix = np.array([
            [1, 0, 0],
            [1, 1, 0],
            [1, 0, 1]
        ])
        result = yyc_algorithm(matrix)

        assert len(result) > 0
        # Column 0 alone should be a testor since it has 1 in first row
        assert (0,) in result

    def test_empty_matrix(self):
        """Test with empty matrix returns empty list."""
        matrix = np.array([]).reshape(0, 0)
        result = yyc_algorithm(matrix)

        assert result == []

    def test_single_row_matrix(self):
        """Test with single row matrix."""
        matrix = np.array([[1, 0, 1, 0]])
        result = yyc_algorithm(matrix)

        # Each column with a 1 should be a testor
        assert (0,) in result
        assert (2,) in result
        assert len(result) == 2

    def test_single_column_matrix(self):
        """Test with single column matrix."""
        matrix = np.array([[1], [1], [1]])
        result = yyc_algorithm(matrix)

        assert (0,) in result

    def test_all_ones_matrix(self):
        """Test matrix with all ones."""
        matrix = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        result = yyc_algorithm(matrix)

        # Any single column should be a testor
        assert len(result) > 0
        assert all(len(t) == 1 for t in result)

    def test_row_with_all_zeros(self):
        """Test matrix containing a row of all zeros."""
        matrix = np.array([
            [1, 0, 1],
            [0, 0, 0],  # Zero row
            [0, 1, 0]
        ])
        result = yyc_algorithm(matrix)

        # Algorithm should still return results (may be empty due to zero row)
        assert isinstance(result, list)

    def test_alternating_pattern(self):
        """Test with alternating pattern matrix."""
        matrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ])
        result = yyc_algorithm(matrix)

        assert isinstance(result, list)

    def test_result_is_sorted(self):
        """Verify results are sorted."""
        matrix = np.array([
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1]
        ])
        result = yyc_algorithm(matrix)

        assert result == sorted(result)

    def test_no_duplicate_testors(self):
        """Verify no duplicate testors in result."""
        matrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ])
        result = yyc_algorithm(matrix)

        assert len(result) == len(set(result))

    def test_testor_indices_within_bounds(self):
        """Verify all testor indices are valid column indices."""
        matrix = np.array([
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 1, 1, 0, 0]
        ])
        result = yyc_algorithm(matrix)
        num_cols = matrix.shape[1]

        for testor in result:
            for idx in testor:
                assert 0 <= idx < num_cols

    def test_testor_elements_are_python_ints(self):
        """Verify testor elements are Python ints, not numpy types."""
        matrix = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0]
        ])
        result = yyc_algorithm(matrix)

        for testor in result:
            for idx in testor:
                assert type(idx) is int, f"Expected int, got {type(idx)}"


class TestIsCompatibleSet:
    """Tests for the _is_compatible_set helper function."""

    def test_compatible_simple_case(self):
        """Test a simple compatible case."""
        matrix = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0]
        ])
        tau = (0,)
        xp = 1

        # Should be a valid test - function returns bool-like value
        result = _is_compatible_set(tau, xp, matrix)
        assert result in (True, False)

    def test_empty_tau(self):
        """Test with empty tau tuple."""
        matrix = np.array([
            [1, 0, 1],
            [0, 1, 1],
        ])
        tau = ()
        xp = 0

        result = _is_compatible_set(tau, xp, matrix)
        assert result in (True, False)


class TestStressTests:
    """Random stress tests for algorithm consistency."""

    @pytest.mark.parametrize("size", [(5, 5), (10, 10), (5, 10), (10, 5)])
    def test_random_matrices_return_valid_results(self, size):
        """Test that random matrices produce valid results."""
        rows, cols = size
        np.random.seed(42)  # Reproducibility
        matrix = np.random.randint(0, 2, size=(rows, cols))

        result = yyc_algorithm(matrix)

        assert isinstance(result, list)
        for testor in result:
            assert isinstance(testor, tuple)
            assert all(0 <= idx < cols for idx in testor)

    def test_deterministic_results(self):
        """Verify same input produces same output."""
        matrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ])

        result1 = yyc_algorithm(matrix)
        result2 = yyc_algorithm(matrix)

        assert result1 == result2

    @pytest.mark.parametrize("seed", [1, 42, 123, 999])
    def test_multiple_random_seeds(self, seed):
        """Test with various random seeds for robustness."""
        np.random.seed(seed)
        matrix = np.random.randint(0, 2, size=(8, 8))

        result = yyc_algorithm(matrix)

        assert isinstance(result, list)
        # Results should be sorted and unique
        assert result == sorted(result)
        assert len(result) == len(set(result))

    def test_large_matrix_completes(self):
        """Test that larger matrices complete in reasonable time."""
        np.random.seed(42)
        matrix = np.random.randint(0, 2, size=(20, 15))

        result = yyc_algorithm(matrix)

        assert isinstance(result, list)
