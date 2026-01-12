"""
Tests for error handling and input validation.

Ensures robust behavior with invalid inputs.
"""

import numpy as np
import pytest

from yyc.core.algorithm import (
    InvalidMatrixError,
    YYCError,
    yyc_algorithm,
)


class TestExceptionHierarchy:
    """Test exception class hierarchy."""

    def test_invalid_matrix_error_is_yyc_error(self):
        """InvalidMatrixError should inherit from YYCError."""
        assert issubclass(InvalidMatrixError, YYCError)

    def test_yyc_error_is_exception(self):
        """YYCError should inherit from Exception."""
        assert issubclass(YYCError, Exception)

    def test_can_catch_specific_error(self):
        """Test catching specific error type."""
        with pytest.raises(InvalidMatrixError):
            yyc_algorithm(np.array([[1, 2, 3]]))

    def test_can_catch_base_error(self):
        """Test catching base YYCError."""
        with pytest.raises(YYCError):
            yyc_algorithm(np.array([[1, 2, 3]]))


class TestInputValidation:
    """Tests for input validation."""

    def test_non_binary_values_raises_error(self):
        """Test that non-binary values raise InvalidMatrixError."""
        matrix = np.array([[1, 2, 0], [0, 1, 3]])

        with pytest.raises(InvalidMatrixError, match="binary"):
            yyc_algorithm(matrix)

    def test_negative_values_raises_error(self):
        """Test that negative values raise InvalidMatrixError."""
        matrix = np.array([[1, -1, 0], [0, 1, 0]])

        with pytest.raises(InvalidMatrixError, match="binary"):
            yyc_algorithm(matrix)

    def test_float_values_raises_error(self):
        """Test that float values raise InvalidMatrixError."""
        matrix = np.array([[1.5, 0, 1], [0, 1, 0]])

        with pytest.raises(InvalidMatrixError, match="binary"):
            yyc_algorithm(matrix)

    def test_1d_array_raises_error(self):
        """Test that 1D array raises InvalidMatrixError."""
        matrix = np.array([1, 0, 1, 0])

        with pytest.raises(InvalidMatrixError, match="2-dimensional"):
            yyc_algorithm(matrix)

    def test_3d_array_raises_error(self):
        """Test that 3D array raises InvalidMatrixError."""
        matrix = np.array([[[1, 0], [0, 1]], [[1, 1], [0, 0]]])

        with pytest.raises(InvalidMatrixError, match="2-dimensional"):
            yyc_algorithm(matrix)

    def test_non_array_input_raises_type_error(self):
        """Test that non-array input raises TypeError."""
        with pytest.raises(TypeError, match="numpy.ndarray"):
            yyc_algorithm("not an array")

        with pytest.raises(TypeError, match="numpy.ndarray"):
            yyc_algorithm(12345)

        with pytest.raises(TypeError, match="numpy.ndarray"):
            yyc_algorithm({"a": 1, "b": 2})


class TestListInputConversion:
    """Tests for automatic list-to-array conversion."""

    def test_list_input_works(self):
        """Test that list input is automatically converted."""
        matrix = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
        result = yyc_algorithm(matrix)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_nested_list_works(self):
        """Test nested list input."""
        matrix = [[1, 0], [0, 1]]
        result = yyc_algorithm(matrix)

        assert isinstance(result, list)

    def test_invalid_list_raises_error(self):
        """Test that invalid list values raise error."""
        matrix = [[1, 0, 2], [0, 1, 0]]

        with pytest.raises(InvalidMatrixError, match="binary"):
            yyc_algorithm(matrix)


class TestEdgeCasesWithValidation:
    """Edge cases that should work despite being unusual."""

    def test_single_element_matrix(self):
        """Test 1x1 matrix."""
        matrix = np.array([[1]])
        result = yyc_algorithm(matrix)

        assert result == [(0,)]

    def test_single_element_zero(self):
        """Test 1x1 matrix with zero."""
        matrix = np.array([[0]])
        result = yyc_algorithm(matrix)

        assert result == []

    def test_very_wide_matrix(self):
        """Test matrix with many more columns than rows."""
        matrix = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])
        result = yyc_algorithm(matrix)

        assert len(result) == 5  # 5 columns with 1s

    def test_very_tall_matrix(self):
        """Test matrix with many more rows than columns."""
        matrix = np.array([[1], [1], [1], [1], [1]])
        result = yyc_algorithm(matrix)

        assert (0,) in result
