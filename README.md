# YYC Algorithm - Typical Testor Finder

A Python implementation of the YYC algorithm for computing **typical testors** from binary matrices. Typical testors are minimal sets of features that can distinguish between all objects in a dataset, making them valuable for feature selection in pattern recognition and machine learning.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-74%20passed-brightgreen.svg)]()

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Python API](#python-api)
- [API Reference](#api-reference)
- [Computational Complexity](#computational-complexity)
- [Testing](#testing)
- [License](#license)

## Background

### What are Testors?

In pattern recognition and data analysis, a **testor** is a subset of features (attributes) that can distinguish between all objects in a dataset. A **typical testor** is a minimal testor—removing any feature from it would result in a set that can no longer distinguish all objects.

### The YYC Algorithm

The YYC algorithm efficiently computes all typical testors from a binary difference matrix. Given a matrix where:
- **Rows** represent pairwise comparisons between objects
- **Columns** represent features/attributes
- A value of **1** indicates the feature distinguishes those two objects
- A value of **0** indicates the feature does not distinguish them

The algorithm iteratively builds testors by:
1. Starting with single-attribute testors from the first row
2. For each subsequent row, either keeping existing testors (if they already cover the row) or extending them with compatible attributes
3. Filtering to maintain only minimal (typical) testors

### Applications

- **Feature Selection**: Identify the minimal set of features needed for classification
- **Dimensionality Reduction**: Reduce dataset complexity while preserving discriminative power
- **Pattern Recognition**: Find optimal attribute combinations for object classification
- **Data Mining**: Discover essential attributes in large datasets

## Installation

### Requirements

- Python 3.8 or higher
- NumPy >= 1.20.0

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/yyc-algorithm.git
cd yyc-algorithm

# Install dependencies
pip install -r requirements.txt

# Install the package (optional, for system-wide use)
pip install -e .
```

### Install Dependencies Only

```bash
pip install numpy>=1.20.0
```

## Quick Start

### Using the CLI

```bash
# Run with a random 5x5 matrix
python main.py --random 5 5

# Run interactively
python main.py --interactive

# Process a matrix file
python main.py --file matrix.txt --verbose
```

### Using the Python API

```python
import numpy as np
from yyc import yyc_algorithm

# Define a binary matrix
matrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1]
])

# Find typical testors
testors = yyc_algorithm(matrix)

print(f"Found {len(testors)} typical testors:")
for testor in testors:
    # Convert to 1-indexed for display
    print(tuple(i + 1 for i in testor))
```

## Usage

### Command Line Interface

The CLI provides several ways to run the algorithm:

```
usage: yyc [-h] [-f FILE] [-r ROWS COLS] [-i] [--zero-indexed] [-v] [--version]

YYC Algorithm - Find typical testors in binary matrices.

options:
  -h, --help            Show this help message and exit
  -f FILE, --file FILE  Path to a file containing the binary matrix
  -r ROWS COLS, --random ROWS COLS
                        Generate a random matrix with specified dimensions
  -i, --interactive     Run in interactive mode
  --zero-indexed        Display testor indices as 0-indexed (default: 1-indexed)
  -v, --verbose         Show additional information including the input matrix
  --version             Show program's version number and exit
```

#### Examples

**Generate and analyze a random matrix:**
```bash
python main.py --random 8 6 --verbose
```

Output:
```
Input matrix:
[1 0 1 0 1 0]
[0 1 0 1 0 1]
[1 1 0 0 1 0]
...

Execution time: 0.000234 seconds
Number of typical testors: 12

Typical testors:
(1, 2, 4)
(1, 3, 6)
...
```

**Load matrix from file:**
```bash
python main.py --file data/sample_matrix.txt
```

Matrix file format (space or comma-separated):
```
1 0 1 0
0 1 0 1
1 1 0 0
0 0 1 1
```

**Interactive mode:**
```bash
python main.py --interactive
```

### Python API

#### Basic Usage

```python
from yyc import yyc_algorithm
import numpy as np

# Using numpy array
matrix = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
testors = yyc_algorithm(matrix)

# Using Python lists (automatically converted)
matrix = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
testors = yyc_algorithm(matrix)
```

#### With Validation

```python
from yyc import yyc_algorithm, validate_matrix, InvalidMatrixError
import numpy as np

matrix = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])

# Validate before processing
is_valid, error = validate_matrix(matrix)
if not is_valid:
    print(f"Invalid matrix: {error}")
else:
    testors = yyc_algorithm(matrix)

# Or use try/except
try:
    testors = yyc_algorithm(matrix)
except InvalidMatrixError as e:
    print(f"Error: {e}")
```

#### Matrix Utilities

```python
from yyc import generate_random_matrix, has_zero_row, validate_matrix

# Generate a random binary matrix
matrix = generate_random_matrix(10, 8)

# Check for problematic zero rows
if has_zero_row(matrix):
    print("Warning: Matrix contains a row of all zeros")

# Validate matrix
is_valid, error = validate_matrix(matrix)
```

## API Reference

### Core Functions

#### `yyc_algorithm(matrix)`

Find all typical testors in a binary matrix.

**Parameters:**
- `matrix` (np.ndarray | List[List[int]]): Binary matrix where rows are objects and columns are attributes

**Returns:**
- `List[Tuple[int, ...]]`: Sorted list of testors (0-indexed column tuples)

**Raises:**
- `InvalidMatrixError`: If matrix contains non-binary values or has invalid dimensions
- `TypeError`: If input is not an array or list

### Utility Functions

#### `generate_random_matrix(rows, cols)`

Generate a random binary matrix.

**Parameters:**
- `rows` (int): Number of rows
- `cols` (int): Number of columns

**Returns:**
- `np.ndarray`: Random binary matrix

#### `validate_matrix(matrix)`

Validate that a matrix is suitable for the YYC algorithm.

**Parameters:**
- `matrix` (np.ndarray): Matrix to validate

**Returns:**
- `Tuple[bool, str]`: (is_valid, error_message)

#### `has_zero_row(matrix)`

Check if matrix contains any row with all zeros.

**Parameters:**
- `matrix` (np.ndarray): Binary matrix

**Returns:**
- `bool`: True if any row is all zeros

### Exceptions

#### `YYCError`

Base exception for all YYC algorithm errors.

#### `InvalidMatrixError`

Raised when the input matrix is invalid (non-binary values, wrong dimensions, etc.).

## Computational Complexity

### Time Complexity

The YYC algorithm has the following complexity characteristics:

- **Best case**: O(n * m) where n = rows, m = columns
  - Occurs when testors are found quickly with minimal branching

- **Worst case**: O(n * m * 2^m)
  - Occurs with highly complex matrices requiring extensive testor exploration

- **Average case**: Depends on matrix structure and density of 1s

### Space Complexity

- O(k * m) where k = number of testors being tracked, m = average testor size

### Performance Tips

1. **Matrix size**: Performance degrades with larger matrices; consider preprocessing to reduce dimensions
2. **Matrix density**: Sparse matrices (fewer 1s) typically process faster
3. **Row ordering**: The algorithm processes rows sequentially; strategic ordering may improve performance

## Benchmarking

A benchmarking tool is included to measure and visualize algorithm performance:

```bash
# Quick benchmark (smaller matrices)
python benchmark.py --quick

# Standard benchmark
python benchmark.py

# Full benchmark (larger matrices)
python benchmark.py --full

# Generate visualization
python benchmark.py --plot --output benchmark.png

# Custom sizes
python benchmark.py --sizes "5x5,10x10,20x15,30x20"
```

### Benchmark Options

```
--sizes SIZES    Custom matrix sizes (e.g., '5x5,10x8,20x12')
--runs RUNS      Runs per size for averaging (default: 3)
--seed SEED      Random seed for reproducibility (default: 42)
--plot           Generate visualization
--output FILE    Save plot to file
--quick          Quick benchmark with smaller sizes
--full           Comprehensive benchmark with more sizes
--quiet          Suppress progress output
```

### Sample Output

```
============================================================
YYC Algorithm Benchmark Results
============================================================

Size         Time (s)     Testors    Density
------------------------------------------------------------
5x5          0.000089     0          44.0%
10x8         0.000606     13         50.0%
15x10        0.001641     31         52.0%
20x12        0.007148     76         51.7%
------------------------------------------------------------
Total time: 0.0095 seconds
============================================================
```

## Testing

Run the test suite with pytest:

```bash
# Run all tests
PYTHONPATH=src pytest tests/ -v

# Run with coverage
PYTHONPATH=src pytest tests/ --cov=yyc --cov-report=html

# Run specific test file
PYTHONPATH=src pytest tests/test_algorithm.py -v
```

### Test Categories

- **Unit tests**: Core algorithm functionality
- **Edge cases**: Empty matrices, single elements, zero rows
- **Stress tests**: Random matrices of various sizes
- **Validation tests**: Error handling and input validation

## Project Structure

```
yyc-algorithm/
├── src/yyc/
│   ├── __init__.py          # Package exports
│   ├── core/
│   │   ├── __init__.py
│   │   └── algorithm.py     # YYC algorithm implementation
│   ├── utils/
│   │   ├── __init__.py
│   │   └── matrix.py        # Matrix utilities
│   ├── cli/
│   │   ├── __init__.py
│   │   └── interface.py     # Command-line interface
│   └── benchmarks/
│       ├── __init__.py
│       └── runner.py        # Benchmark runner and plotting
├── tests/
│   ├── test_algorithm.py    # Algorithm tests
│   ├── test_matrix.py       # Utility tests
│   └── test_error_handling.py
├── examples/
│   └── sample_matrix.txt    # Sample matrix file
├── main.py                  # Main entry point
├── benchmark.py             # Benchmark script
├── pyproject.toml           # Package configuration
├── requirements.txt         # Dependencies
└── README.md
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- Alba-Cabrera, E., Ibarra-Fiallo, J., Godoy-Calderon, S., & Cervantes-Alonso, F. (2014). YYC: A fast performance incremental algorithm for finding typical testors. In *Progress in Pattern Recognition, Image Analysis, Computer Vision, and Applications (CIARP 2014)* (Lecture Notes in Computer Science, Vol. 8827, pp. 416–423). Springer. https://doi.org/10.1007/978-3-319-12568-8_51
- Ruiz-Shulcloper, J., & Lazo-Cortés, M. (1995). Mathematical algorithms for the supervised classification based on the typical testors concept.
- Santiesteban, Y., & Pons-Porrata, A. (2003). LEX: A new algorithm for the calculation of typical testors. *Revista Ciencias Matemáticas*, 21(1), 85-95.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
