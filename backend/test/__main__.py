"""Run offline regressions with: python -m backend.test."""
from pathlib import Path
import sys
import unittest


def main():
    directory = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(directory), pattern='test_*.py', top_level_dir=str(directory.parents[1]))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
