"""
YYC Algorithm Package.

A Python implementation of the YYC algorithm for finding typical testors
in binary matrices, useful for feature selection in pattern recognition.
"""

from yyc.core.algorithm import yyc_algorithm
from yyc.utils.matrix import (
    generate_random_matrix,
    has_zero_row,
    parse_matrix_from_list,
    validate_matrix,
)

__version__ = "1.0.0"
__all__ = [
    "yyc_algorithm",
    "generate_random_matrix",
    "validate_matrix",
    "has_zero_row",
    "parse_matrix_from_list",
]
