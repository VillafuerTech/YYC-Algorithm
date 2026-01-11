#!/usr/bin/env python3
"""
Main entry point for the YYC Algorithm CLI.

Usage:
    python main.py [options]
    python main.py --help
"""

import sys

from yyc.cli.interface import main

if __name__ == "__main__":
    sys.exit(main())
