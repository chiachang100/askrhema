# How to debug AskRhema's Streamlit app and view logs:

## 1. Streamlit Built-in Debugging Options

### Run with Detailed Logging
```bash
# Run with debug logging level
uv run streamlit run app.py --logger.level=debug

# Or with info level (less verbose)
uv run streamlit run app.py --logger.level=info

# Run with verbose mode
uv run streamlit run app.py --verbose
```

### Run with Server Details
```bash
# Show all server logs
uv run streamlit run app.py --server.runOnSave false --server.enableCORS false --server.enableXsrfProtection false
```

## 2. Add Python Logging to Your App

### Update `app.py` to include proper logging:

```python
# app.py (add logging configuration at the top)
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        #logging.StreamHandler(sys.stdout),
        logging.StreamHandler(),
        logging.FileHandler('askrhema_debug.log', encoding="utf-8")  # Also log to file
    ]
)
logger = logging.getLogger(__name__)

# Then in your functions, add log statements
def handle_search(...):
    logger.info(f"Search started: query='{query}', top_k={top_k}")
    logger.debug(f"Filters: book={book_filter}, testament={testament_filter}")
    try:
        results = search_engine.search(...)
        logger.info(f"Search completed: found {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Search failed: {str(e)}", exc_info=True)
        raise
```

## 3. Create a Debug Configuration File

### Create `debug_config.py`:

```python
# debug_config.py
"""Debug configuration for AskRhema."""

import logging
import os
from typing import Optional

def setup_debug_logging(level: str = "DEBUG", log_file: Optional[str] = "askrhema_debug.log"):
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper(), logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = None
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)
    
    # Set specific levels for noisy modules
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return root_logger

# Enable debug mode via environment variable
DEBUG_MODE = os.getenv("SEEKRHEMA_DEBUG", "false").lower() == "true"

if DEBUG_MODE:
    setup_debug_logging("DEBUG")
else:
    setup_debug_logging("INFO")
```

## 4. Add Streamlit Session State Debugging

### Add a debug section to your sidebar:

```python
# In app.py, add to sidebar
def display_sidebar():
    # ... existing code ...
    
    st.sidebar.divider()
    
    # Debug section
    if st.sidebar.checkbox("🐛 Debug Mode", help="Show debug information"):
        st.sidebar.subheader("🔧 Debug Info")
        
        # Show session state keys
        with st.sidebar.expander("Session State"):
            for key, value in st.session_state.items():
                if key not in ["search_engine", "search_results"]:  # Skip large objects
                    st.sidebar.write(f"**{key}**: {value}")
        
        # Show search engine status
        if st.session_state.search_engine:
            st.sidebar.success("✅ Search Engine: Initialized")
            st.sidebar.write(f"Collection: {st.session_state.search_engine.config.collection_name}")
            st.sidebar.write(f"Vector Size: {st.session_state.search_engine.config.vector_size}")
        else:
            st.sidebar.error("❌ Search Engine: Not Initialized")
        
        # Show environment info
        with st.sidebar.expander("Environment"):
            import sys
            import platform
            st.sidebar.write(f"Python: {sys.version}")
            st.sidebar.write(f"Platform: {platform.platform()}")
            st.sidebar.write(f"Streamlit: {st.__version__}")
```

## 5. Use Streamlit's Built-in Profiler
- Run with profiler
```bash
uv run streamlit run app.py --server.runOnSave false
```

- Then in your app, you can add performance measurements
```python
import time

def handle_search(...):
    start_time = time.time()
    # ... search logic ...
    elapsed = time.time() - start_time
    logger.info(f"Search took {elapsed:.3f} seconds")
    st.sidebar.write(f"⏱️ Search time: {elapsed:.3f}s")  # Show in UI
```

## 6. Create a Debug Helper Function
```python
# debug_utils.py
"""Debug utilities for AskRhema."""

import logging
import time
from functools import wraps
from typing import Any, Callable
import streamlit as st

logger = logging.getLogger(__name__)

def log_execution_time(func: Callable) -> Callable:
    """Decorator to log execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

def debug_print(*args, **kwargs):
    """Print debug messages only if debug mode is enabled."""
    if st.session_state.get("debug_mode", False):
        print("[DEBUG]", *args, **kwargs)

# Use in your code
@log_execution_time
def search_function(query):
    # Your search logic
    pass
```

## 7. View Logs in Real-Time
- On Windows (PowerShell):
```powershell
# View logs in real-time
Get-Content askrhema_debug.log -Wait

# Or using tail (if installed)
tail -f askrhema_debug.log
```

- On Linux/Mac:
```bash
# View logs in real-time
tail -f askrhema_debug.log

# Or with color
tail -f askrhema_debug.log | grep --color=auto -E "ERROR|WARNING|INFO|DEBUG"
```

## 8. Complete Debug Configuration Setup

### Create a .env file for debug settings:

```bash
# .env
SEEKRHEMA_DEBUG=true
SEEKRHEMA_LOG_LEVEL=DEBUG
SEEKRHEMA_LOG_FILE=askrhema_debug.log
```

Then in your code:

```python
# app.py (at the top)
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file

DEBUG_MODE = os.getenv("SEEKRHEMA_DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("SEEKRHEMA_LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("SEEKRHEMA_LOG_FILE", "askrhema_debug.log")
```

## 9. Quick Debugging Commands
```bash
# Run with maximum debug output
uv run streamlit run app.py --logger.level=debug 2>&1 | tee streamlit.log

# Run and save all output to a file
uv run streamlit run app.py > app.log 2>&1

# Run with Python debugger (pdb)
uv run python -m pdb -c "import streamlit.cli; streamlit.cli.main()" run app.py

# Run with Python's trace module
uv run python -m trace --trace app.py
```

## 10. Streamlit Developer Mode

### Enable Streamlit's developer mode:

```bash
# Set environment variable
export STREAMLIT_DEV_MODE=true

# Or on Windows
set STREAMLIT_DEV_MODE=true

# Then run
uv run streamlit run app.py
```

## Recommended Debugging Workflow

1. Start with info-level logging:

```bash
uv run streamlit run app.py --logger.level=info
```

2. If you see errors, switch to debug:

```bash
uv run streamlit run app.py --logger.level=debug
```

3. Add logging to specific functions:

```python
logger.info(f"Search query: {query}")
logger.debug(f"Search parameters: {locals()}")
```

4. Use the sidebar debug mode to inspect session state

5. Check the log file for persistent issues:

```bash
tail -f askrhema_debug.log
```

This setup gives you comprehensive debugging capabilities for your AskRhema application!
---
