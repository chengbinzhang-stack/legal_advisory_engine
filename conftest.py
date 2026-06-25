"""Project-level pytest configuration."""
import sys
import os

# Make `src` importable when running `pytest` from the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
