import pathlib
import sys

# Run the tests from a fresh clone without installing the package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
