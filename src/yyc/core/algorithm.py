"""
YYC Algorithm for computing typical testors from a binary matrix.

This module implements the YYC algorithm for feature selection through
the identification of typical testors - minimal sets of attributes that
can distinguish between all objects in a dataset.
"""

from typing import List, Set, Tuple, Union

import numpy as np


class YYCError(Exception):
    """Base exception for YYC algorithm errors."""

    pass


class InvalidMatrixError(YYCError):
    """Raised when the input matrix is invalid."""

    pass


def _validate_input(matrix: np.ndarray) -> None:
    """
    Validate the input matrix for the YYC algorithm.

    Args:
        matrix: The matrix to validate.

    Raises:
        InvalidMatrixError: If the matrix is invalid.
        TypeError: If the input is not a numpy array.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError(
            f"Expected numpy.ndarray, got {type(matrix).__name__}. "
            "Use numpy.array() to convert your data."
        )

    if matrix.ndim != 2:
        raise InvalidMatrixError(
            f"Matrix must be 2-dimensional, got {matrix.ndim} dimensions."
        )

    if matrix.size > 0 and not np.all((matrix == 0) | (matrix == 1)):
        raise InvalidMatrixError(
            "Matrix must contain only binary values (0 and 1)."
        )


def _is_compatible_set(
    tau: Tuple[int, ...],
    xp: int,
    matrix: np.ndarray
) -> bool:
    """
    Check if adding attribute xp to testor tau maintains compatibility.

    A set is compatible if:
    1. At least (len(tau) + 1) rows have exactly one 1 in the selected attributes
    2. All columns have at least one 1 across the filtered rows

    Args:
        tau: Current testor (tuple of column indices).
        xp: Candidate attribute index to add.
        matrix: Binary matrix as numpy array.

    Returns:
        True if the extended set is compatible, False otherwise.
    """
    indices = list(tau) + [xp]
    submatrix = matrix[:, indices]

    row_sums = submatrix.sum(axis=1)
    rows_with_one = row_sums == 1

    condition1 = rows_with_one.sum() >= len(tau) + 1

    if not condition1:
        return False

    filtered_rows = submatrix[rows_with_one]

    if filtered_rows.size == 0:
        return False

    condition2 = np.all(filtered_rows.sum(axis=0) >= 1)

    return condition1 and condition2


def yyc_algorithm(
    matrix: Union[np.ndarray, List[List[int]]]
) -> List[Tuple[int, ...]]:
    """
    Find all typical testors in a binary matrix using the YYC algorithm.

    A typical testor is a minimal set of attributes (columns) that can
    distinguish between all objects (rows) in the matrix.

    Args:
        matrix: A binary (0/1) numpy array or list of lists where rows
                represent objects and columns represent attributes.

    Returns:
        A sorted list of testors, where each testor is a tuple of
        column indices (0-indexed).

    Raises:
        InvalidMatrixError: If the matrix contains non-binary values or
                           has invalid dimensions.
        TypeError: If the input is not a numpy array or list.

    Example:
        >>> import numpy as np
        >>> matrix = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        >>> testors = yyc_algorithm(matrix)
        >>> print(testors)
    """
    # Convert list input to numpy array
    if isinstance(matrix, list):
        try:
            matrix = np.array(matrix)
        except (ValueError, TypeError) as e:
            raise InvalidMatrixError(f"Could not convert input to array: {e}")

    _validate_input(matrix)

    if matrix.size == 0:
        return []

    psi_star: Set[Tuple[int, ...]] = set()

    first_row = matrix[0]
    for col_idx in range(len(first_row)):
        if first_row[col_idx] == 1:
            psi_star.add((col_idx,))

    for row in matrix[1:]:
        psi_aux: Set[Tuple[int, ...]] = set()

        for tau in psi_star:
            ones_in_row = np.where(row == 1)[0]

            if any(row[idx] == 1 for idx in tau):
                psi_aux.add(tuple(sorted(tau)))
            else:
                for xp in ones_in_row:
                    if _is_compatible_set(tau, int(xp), matrix):
                        psi_aux.add(tuple(sorted(tau + (int(xp),))))

        psi_star = psi_aux

    return sorted(set(psi_star))
