"""
Unit tests for matrix utilities.

Tests cover:
- Matrix generation
- Validation functions
- Parsing and formatting
- Edge cases and error conditions
"""

import numpy as np
import pytest

from yyc.utils.matrix import (
    generate_random_matrix,
    validate_matrix,
    has_zero_row,
    format_matrix,
    parse_matrix_from_list,
)


class TestGenerateRandomMatrix:
    """Tests for generate_random_matrix function."""

    def test_correct_dimensions(self):
        """Test that generated matrix has correct dimensions."""
        matrix = generate_random_matrix(5, 10)

        assert matrix.shape == (5, 10)

    def test_binary_values_only(self):
        """Test that matrix contains only 0s and 1s."""
        matrix = generate_random_matrix(10, 10)

        assert np.all((matrix == 0) | (matrix == 1))

    def test_returns_numpy_array(self):
        """Test that function returns numpy array."""
        matrix = generate_random_matrix(3, 3)

        assert isinstance(matrix, np.ndarray)

    def test_different_sizes(self):
        """Test various matrix sizes."""
        for rows in [1, 5, 10, 50]:
            for cols in [1, 5, 10, 50]:
                matrix = generate_random_matrix(rows, cols)
                assert matrix.shape == (rows, cols)

    def test_zero_rows_raises_error(self):
        """Test that zero rows raises ValueError."""
        with pytest.raises(ValueError, match="positive integers"):
            generate_random_matrix(0, 5)

    def test_zero_cols_raises_error(self):
        """Test that zero columns raises ValueError."""
        with pytest.raises(ValueError, match="positive integers"):
            generate_random_matrix(5, 0)

    def test_negative_dimensions_raises_error(self):
        """Test that negative dimensions raise ValueError."""
        with pytest.raises(ValueError, match="positive integers"):
            generate_random_matrix(-1, 5)

        with pytest.raises(ValueError, match="positive integers"):
            generate_random_matrix(5, -1)


class TestValidateMatrix:
    """Tests for validate_matrix function."""

    def test_valid_binary_matrix(self):
        """Test that valid binary matrix passes validation."""
        matrix = np.array([[1, 0, 1], [0, 1, 0]])
        is_valid, error = validate_matrix(matrix)

        assert is_valid is True
        assert error == ""

    def test_empty_matrix_fails(self):
        """Test that empty matrix fails validation."""
        matrix = np.array([]).reshape(0, 0)
        is_valid, error = validate_matrix(matrix)

        assert is_valid is False
        assert "empty" in error.lower()

    def test_non_binary_values_fail(self):
        """Test that non-binary values fail validation."""
        matrix = np.array([[1, 0, 2], [0, 1, 0]])
        is_valid, error = validate_matrix(matrix)

        assert is_valid is False
        assert "binary" in error.lower()

    def test_negative_values_fail(self):
        """Test that negative values fail validation."""
        matrix = np.array([[1, 0, -1], [0, 1, 0]])
        is_valid, error = validate_matrix(matrix)

        assert is_valid is False
        assert "binary" in error.lower()

    def test_float_values_fail(self):
        """Test that float values fail validation."""
        matrix = np.array([[1.0, 0.5, 1.0], [0.0, 1.0, 0.0]])
        is_valid, error = validate_matrix(matrix)

        assert is_valid is False
        assert "binary" in error.lower()

    def test_all_zeros_valid(self):
        """Test that all-zeros matrix is valid (though unusual)."""
        matrix = np.array([[0, 0, 0], [0, 0, 0]])
        is_valid, error = validate_matrix(matrix)

        assert is_valid is True

    def test_all_ones_valid(self):
        """Test that all-ones matrix is valid."""
        matrix = np.array([[1, 1, 1], [1, 1, 1]])
        is_valid, error = validate_matrix(matrix)

        assert is_valid is True


class TestHasZeroRow:
    """Tests for has_zero_row function."""

    def test_matrix_with_zero_row(self):
        """Test detection of zero row."""
        matrix = np.array([
            [1, 0, 1],
            [0, 0, 0],  # Zero row
            [1, 1, 0]
        ])

        assert has_zero_row(matrix) is True

    def test_matrix_without_zero_row(self):
        """Test matrix without zero rows."""
        matrix = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0]
        ])

        assert has_zero_row(matrix) is False

    def test_all_zeros_matrix(self):
        """Test matrix with all zeros."""
        matrix = np.array([
            [0, 0, 0],
            [0, 0, 0]
        ])

        assert has_zero_row(matrix) is True

    def test_single_row_all_zeros(self):
        """Test single row of all zeros."""
        matrix = np.array([[0, 0, 0, 0]])

        assert has_zero_row(matrix) is True

    def test_single_row_with_one(self):
        """Test single row with at least one 1."""
        matrix = np.array([[0, 1, 0, 0]])

        assert has_zero_row(matrix) is False

    def test_first_row_zero(self):
        """Test zero row at beginning."""
        matrix = np.array([
            [0, 0, 0],
            [1, 1, 1]
        ])

        assert has_zero_row(matrix) is True

    def test_last_row_zero(self):
        """Test zero row at end."""
        matrix = np.array([
            [1, 1, 1],
            [0, 0, 0]
        ])

        assert has_zero_row(matrix) is True


class TestFormatMatrix:
    """Tests for format_matrix function."""

    def test_basic_formatting(self):
        """Test basic matrix formatting."""
        matrix = np.array([[1, 0], [0, 1]])
        result = format_matrix(matrix)

        assert "[1 0]" in result
        assert "[0 1]" in result

    def test_single_row(self):
        """Test formatting single row matrix."""
        matrix = np.array([[1, 0, 1, 0]])
        result = format_matrix(matrix)

        assert result == "[1 0 1 0]"

    def test_multiline_output(self):
        """Test that multi-row matrices produce multiline output."""
        matrix = np.array([[1, 0], [0, 1], [1, 1]])
        result = format_matrix(matrix)
        lines = result.split("\n")

        assert len(lines) == 3


class TestParseMatrixFromList:
    """Tests for parse_matrix_from_list function."""

    def test_valid_list_parsing(self):
        """Test parsing valid list of lists."""
        data = [[1, 0, 1], [0, 1, 0]]
        matrix = parse_matrix_from_list(data)

        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (2, 3)
        np.testing.assert_array_equal(matrix, np.array([[1, 0, 1], [0, 1, 0]]))

    def test_empty_list_raises_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            parse_matrix_from_list([])

    def test_jagged_array_raises_error(self):
        """Test that jagged array raises ValueError."""
        data = [[1, 0, 1], [0, 1]]  # Different lengths
        with pytest.raises(ValueError, match="[Jj]agged"):
            parse_matrix_from_list(data)

    def test_single_row(self):
        """Test parsing single row."""
        data = [[1, 0, 1, 0, 1]]
        matrix = parse_matrix_from_list(data)

        assert matrix.shape == (1, 5)

    def test_single_column(self):
        """Test parsing single column."""
        data = [[1], [0], [1]]
        matrix = parse_matrix_from_list(data)

        assert matrix.shape == (3, 1)

    def test_int8_dtype(self):
        """Test that result has int8 dtype for memory efficiency."""
        data = [[1, 0], [0, 1]]
        matrix = parse_matrix_from_list(data)

        assert matrix.dtype == np.int8


class TestIntegration:
    """Integration tests combining multiple utilities."""

    def test_generate_validate_cycle(self):
        """Test that generated matrices pass validation."""
        for _ in range(10):
            matrix = generate_random_matrix(5, 5)
            is_valid, error = validate_matrix(matrix)

            assert is_valid is True
            assert error == ""

    def test_parse_format_roundtrip(self):
        """Test that parse and format are consistent."""
        original = [[1, 0, 1], [0, 1, 0], [1, 1, 1]]
        matrix = parse_matrix_from_list(original)
        formatted = format_matrix(matrix)

        # Verify the formatted output contains expected values
        assert "[1 0 1]" in formatted
        assert "[0 1 0]" in formatted
        assert "[1 1 1]" in formatted
