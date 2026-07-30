# tests/conftest.py
"""Pytest configuration file."""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Also add the project root to PYTHONPATH environment variable
os.environ["PYTHONPATH"] = str(project_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

# This ensures that the engine module can be imported
import engine