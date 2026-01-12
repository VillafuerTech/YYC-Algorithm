#!/usr/bin/env python3
"""
Benchmark script for the YYC Algorithm.

Generates random matrices of increasing sizes and measures execution time
to demonstrate the algorithm's performance characteristics.

Usage:
    python benchmark.py [options]
    python benchmark.py --help
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from yyc.benchmarks.runner import run_benchmark, plot_results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="Benchmark the YYC algorithm with various matrix sizes.",
        epilog="Example: python benchmark.py --plot --output benchmark.png",
    )

    parser.add_argument(
        "--sizes",
        type=str,
        default=None,
        help=(
            "Comma-separated list of sizes as ROWSxCOLS "
            "(e.g., '5x5,10x8,20x12'). Defaults to standard progression."
        ),
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per size for averaging (default: 3)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a visualization of the results",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Save plot to file (e.g., benchmark.png)",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (useful for saving only)",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick benchmark with smaller sizes",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run a comprehensive benchmark with more sizes",
    )

    return parser.parse_args()


def parse_sizes(sizes_str: str) -> list:
    """Parse sizes string into list of (rows, cols) tuples."""
    sizes = []
    for size in sizes_str.split(","):
        size = size.strip()
        if "x" in size:
            rows, cols = size.split("x")
            sizes.append((int(rows), int(cols)))
        else:
            raise ValueError(f"Invalid size format: {size}. Use ROWSxCOLS.")
    return sizes


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Determine sizes to benchmark
    if args.sizes:
        try:
            sizes = parse_sizes(args.sizes)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    elif args.quick:
        sizes = [
            (5, 5),
            (10, 8),
            (15, 10),
            (20, 12),
        ]
    elif args.full:
        sizes = [
            (5, 5),
            (10, 8),
            (15, 10),
            (20, 12),
            (25, 15),
            (30, 15),
            (35, 18),
            (40, 18),
            (50, 20),
            (60, 22),
            (75, 25),
        ]
    else:
        sizes = None  # Use defaults

    # Run benchmark
    suite = run_benchmark(
        sizes=sizes,
        runs_per_size=args.runs,
        seed=args.seed,
        verbose=not args.quiet,
    )

    # Print summary
    if not args.quiet:
        print()
        print(suite.summary())

    # Generate plot if requested
    if args.plot or args.output:
        plot_results(
            suite,
            output_path=args.output,
            show=not args.no_show and args.output is None,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
