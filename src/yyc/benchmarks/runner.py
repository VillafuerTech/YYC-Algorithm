"""
Benchmark runner for the YYC algorithm.

Measures execution time across various matrix sizes to demonstrate
algorithm performance characteristics.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from yyc.core.algorithm import yyc_algorithm
from yyc.utils.matrix import generate_random_matrix


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    rows: int
    cols: int
    execution_time: float
    num_testors: int
    matrix_density: float

    def __str__(self) -> str:
        return (
            f"{self.rows}x{self.cols}: {self.execution_time:.6f}s "
            f"({self.num_testors} testors, {self.matrix_density:.1%} density)"
        )


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    results: List[BenchmarkResult] = field(default_factory=list)
    total_time: float = 0.0

    def add(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)
        self.total_time += result.execution_time

    def get_sizes(self) -> List[Tuple[int, int]]:
        """Get list of (rows, cols) tuples."""
        return [(r.rows, r.cols) for r in self.results]

    def get_times(self) -> List[float]:
        """Get list of execution times."""
        return [r.execution_time for r in self.results]

    def get_testor_counts(self) -> List[int]:
        """Get list of testor counts."""
        return [r.num_testors for r in self.results]

    def summary(self) -> str:
        """Generate a summary of the benchmark suite."""
        if not self.results:
            return "No benchmark results."

        lines = [
            "=" * 60,
            "YYC Algorithm Benchmark Results",
            "=" * 60,
            "",
            f"{'Size':<12} {'Time (s)':<12} {'Testors':<10} {'Density':<10}",
            "-" * 60,
        ]

        for r in self.results:
            lines.append(
                f"{r.rows}x{r.cols:<7} {r.execution_time:<12.6f} "
                f"{r.num_testors:<10} {r.matrix_density:<.1%}"
            )

        lines.extend([
            "-" * 60,
            f"Total time: {self.total_time:.4f} seconds",
            f"Benchmarks run: {len(self.results)}",
            "=" * 60,
        ])

        return "\n".join(lines)


def run_benchmark(
    sizes: Optional[List[Tuple[int, int]]] = None,
    runs_per_size: int = 3,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> BenchmarkSuite:
    """
    Run benchmarks across various matrix sizes.

    Args:
        sizes: List of (rows, cols) tuples to benchmark.
               Defaults to a standard progression.
        runs_per_size: Number of runs per size (results are averaged).
        seed: Random seed for reproducibility.
        verbose: Print progress during benchmarking.

    Returns:
        BenchmarkSuite containing all results.
    """
    if sizes is None:
        sizes = [
            (5, 5),
            (10, 8),
            (15, 10),
            (20, 12),
            (25, 15),
            (30, 15),
            (40, 18),
            (50, 20),
        ]

    if seed is not None:
        np.random.seed(seed)

    suite = BenchmarkSuite()

    if verbose:
        print("Starting YYC Algorithm Benchmark")
        print(f"Testing {len(sizes)} matrix sizes, {runs_per_size} runs each")
        print("-" * 50)

    for rows, cols in sizes:
        times = []
        testors_count = 0
        density = 0.0

        for run in range(runs_per_size):
            matrix = generate_random_matrix(rows, cols)
            density = np.mean(matrix)

            start = time.perf_counter()
            result = yyc_algorithm(matrix)
            elapsed = time.perf_counter() - start

            times.append(elapsed)
            testors_count = len(result)

        avg_time = np.mean(times)

        benchmark = BenchmarkResult(
            rows=rows,
            cols=cols,
            execution_time=avg_time,
            num_testors=testors_count,
            matrix_density=density,
        )

        suite.add(benchmark)

        if verbose:
            print(f"  {benchmark}")

    if verbose:
        print("-" * 50)
        print(f"Benchmark complete. Total time: {suite.total_time:.4f}s")

    return suite


def plot_results(
    suite: BenchmarkSuite,
    output_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot benchmark results.

    Args:
        suite: BenchmarkSuite with results to plot.
        output_path: Path to save the plot (optional).
        show: Whether to display the plot interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install with:")
        print("  pip install matplotlib")
        return

    if not suite.results:
        print("No results to plot.")
        return

    sizes = suite.get_sizes()
    times = suite.get_times()
    testors = suite.get_testor_counts()

    # Create size labels
    labels = [f"{r}x{c}" for r, c in sizes]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Execution Time
    color1 = "#2ecc71"
    ax1.bar(labels, times, color=color1, alpha=0.8, edgecolor="black")
    ax1.set_xlabel("Matrix Size (rows x cols)", fontsize=11)
    ax1.set_ylabel("Execution Time (seconds)", fontsize=11)
    ax1.set_title("YYC Algorithm Performance", fontsize=13, fontweight="bold")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3)

    # Add time labels on bars
    for i, (label, t) in enumerate(zip(labels, times)):
        ax1.annotate(
            f"{t:.4f}s",
            xy=(i, t),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Plot 2: Testors Found
    color2 = "#3498db"
    ax2.bar(labels, testors, color=color2, alpha=0.8, edgecolor="black")
    ax2.set_xlabel("Matrix Size (rows x cols)", fontsize=11)
    ax2.set_ylabel("Number of Typical Testors", fontsize=11)
    ax2.set_title("Typical Testors Found", fontsize=13, fontweight="bold")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)

    # Add count labels on bars
    for i, (label, count) in enumerate(zip(labels, testors)):
        ax2.annotate(
            str(count),
            xy=(i, count),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()
