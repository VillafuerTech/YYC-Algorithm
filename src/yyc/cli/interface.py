"""
Command Line Interface for the YYC Algorithm.

Provides a professional CLI for finding typical testors in binary matrices.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from yyc.core.algorithm import yyc_algorithm
from yyc.utils.matrix import (
    format_matrix,
    generate_random_matrix,
    has_zero_row,
    parse_matrix_from_list,
    validate_matrix,
)


def load_matrix_from_file(filepath: Path) -> np.ndarray:
    """
    Load a binary matrix from a file.

    Supports space-separated or comma-separated values.

    Args:
        filepath: Path to the matrix file.

    Returns:
        Numpy array containing the matrix.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is invalid.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    data = []
    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "," in line:
                values = line.split(",")
            else:
                values = line.split()

            try:
                row = [int(v.strip()) for v in values]
                data.append(row)
            except ValueError as e:
                raise ValueError(
                    f"Invalid value on line {line_num}: {e}"
                ) from e

    if not data:
        raise ValueError("File contains no valid matrix data.")

    return parse_matrix_from_list(data)


def format_testors(testors: list, one_indexed: bool = True) -> str:
    """
    Format testors for display.

    Args:
        testors: List of testor tuples.
        one_indexed: If True, display 1-indexed column numbers.

    Returns:
        Formatted string representation.
    """
    lines = []
    for testor in testors:
        if one_indexed:
            display = tuple(x + 1 for x in testor)
        else:
            display = testor
        lines.append(str(display))
    return "\n".join(lines)


def run_interactive_mode() -> None:
    """Run the interactive menu-based interface."""
    print("\n=== YYC Algorithm - Typical Testor Finder ===\n")

    while True:
        print("\n1. Generate a random matrix")
        print("2. Enter a custom matrix")
        print("3. Exit")

        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            try:
                rows = int(input("Number of rows (default 5): ").strip() or "5")
                cols = int(input("Number of columns (default 5): ").strip() or "5")
            except ValueError:
                print("Error: Please enter valid integers.")
                continue

            matrix = generate_random_matrix(rows, cols)
            print("\nGenerated matrix:")
            print(format_matrix(matrix))

        elif choice == "2":
            print("Enter the matrix (one row per line, space-separated):")
            print("(Press Enter twice to finish)")

            data = []
            while True:
                row_input = input().strip()
                if not row_input:
                    break
                try:
                    row = [int(x) for x in row_input.split()]
                    data.append(row)
                except ValueError:
                    print("Error: Invalid input. Use space-separated integers.")
                    continue

            if not data:
                print("Error: No matrix entered.")
                continue

            try:
                matrix = parse_matrix_from_list(data)
            except ValueError as e:
                print(f"Error: {e}")
                continue

            print("\nEntered matrix:")
            print(format_matrix(matrix))

        elif choice == "3":
            print("Goodbye!")
            break

        else:
            print("Invalid option. Please choose from the menu.")
            continue

        is_valid, error = validate_matrix(matrix)
        if not is_valid:
            print(f"Error: {error}")
            continue

        start_time = time.perf_counter()
        result = yyc_algorithm(matrix)
        elapsed = time.perf_counter() - start_time

        print(f"\nExecution time: {elapsed:.6f} seconds")
        print(f"\nNumber of typical testors: {len(result)}")
        print("\nTypical testors (1-indexed):")
        print(format_testors(result))

        if has_zero_row(matrix):
            print("\nWarning: The matrix contains a row of all zeros.")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="yyc",
        description="YYC Algorithm - Find typical testors in binary matrices.",
        epilog="Example: yyc --file matrix.txt",
    )

    parser.add_argument(
        "-f", "--file",
        type=Path,
        help="Path to a file containing the binary matrix.",
    )

    parser.add_argument(
        "-r", "--random",
        nargs=2,
        type=int,
        metavar=("ROWS", "COLS"),
        help="Generate a random matrix with specified dimensions.",
    )

    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode.",
    )

    parser.add_argument(
        "--zero-indexed",
        action="store_true",
        help="Display testor indices as 0-indexed (default is 1-indexed).",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show additional information including the input matrix.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    return parser


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: Command line arguments (defaults to sys.argv).

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    parsed = parser.parse_args(args)

    if parsed.interactive:
        run_interactive_mode()
        return 0

    matrix: Optional[np.ndarray] = None

    if parsed.file:
        try:
            matrix = load_matrix_from_file(parsed.file)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif parsed.random:
        rows, cols = parsed.random
        if rows <= 0 or cols <= 0:
            print("Error: Dimensions must be positive integers.", file=sys.stderr)
            return 1
        matrix = generate_random_matrix(rows, cols)

    else:
        run_interactive_mode()
        return 0

    is_valid, error = validate_matrix(matrix)
    if not is_valid:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if parsed.verbose:
        print("Input matrix:")
        print(format_matrix(matrix))
        print()

    start_time = time.perf_counter()
    result = yyc_algorithm(matrix)
    elapsed = time.perf_counter() - start_time

    one_indexed = not parsed.zero_indexed

    print(f"Execution time: {elapsed:.6f} seconds")
    print(f"Number of typical testors: {len(result)}")
    print("\nTypical testors:")
    print(format_testors(result, one_indexed=one_indexed))

    if has_zero_row(matrix):
        print("\nWarning: The matrix contains a row of all zeros.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
