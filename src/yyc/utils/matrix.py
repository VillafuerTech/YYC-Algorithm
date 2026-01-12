"""
Matrix utilities for the YYC algorithm.

Provides functions for generating, validating, and formatting binary matrices.
"""

from typing import List, Tuple

import numpy as np


def generate_random_matrix(rows: int, cols: int) -> np.ndarray:
    """
    Generate a random binary matrix.

    Args:
        rows: Number of rows.
        cols: Number of columns.

    Returns:
        A numpy array of shape (rows, cols) with random 0s and 1s.

    Raises:
        ValueError: If rows or cols are not positive integers.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("Rows and columns must be positive integers.")

    return np.random.randint(0, 2, size=(rows, cols))


def validate_matrix(matrix: np.ndarray) -> Tuple[bool, str]:
    """
    Validate that a matrix is suitable for the YYC algorithm.

    Checks:
    - Matrix is not empty
    - Matrix contains only binary values (0 and 1)
    - Matrix is not jagged (all rows have same length)

    Args:
        matrix: The matrix to validate.

    Returns:
        A tuple of (is_valid, error_message).
        If valid, error_message is empty string.
    """
    if matrix.size == 0:
        return False, "Matrix is empty."

    if not np.all((matrix == 0) | (matrix == 1)):
        return False, "Matrix must contain only binary values (0 and 1)."

    return True, ""


def has_zero_row(matrix: np.ndarray) -> bool:
    """
    Check if the matrix contains any row with all zeros.

    A zero row means no features distinguish that object, which may
    indicate an issue with the input data.

    Args:
        matrix: Binary matrix as numpy array.

    Returns:
        True if any row is all zeros, False otherwise.
    """
    return bool(np.any(np.all(matrix == 0, axis=1)))


def format_matrix(matrix: np.ndarray) -> str:
    """
    Format a matrix for display.

    Args:
        matrix: The matrix to format.

    Returns:
        A string representation of the matrix.
    """
    lines = []
    for row in matrix:
        lines.append("[" + " ".join(str(int(val)) for val in row) + "]")
    return "\n".join(lines)


def parse_matrix_from_list(data: List[List[int]]) -> np.ndarray:
    """
    Convert a list of lists to a numpy array.

    Args:
        data: List of lists containing integer values.

    Returns:
        Numpy array representation of the data.

    Raises:
        ValueError: If the input is jagged or contains non-integer values.
    """
    if not data:
        raise ValueError("Input list is empty.")

    row_lengths = [len(row) for row in data]
    if len(set(row_lengths)) > 1:
        raise ValueError("Jagged arrays are not supported. All rows must have the same length.")

    return np.array(data, dtype=np.int8)
