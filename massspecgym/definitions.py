"""Global variables used across the package."""
import pathlib

# Dirs
MASSSPECGYM_ROOT_DIR = pathlib.Path(__file__).parent.absolute()
MASSSPECGYM_REPO_DIR = MASSSPECGYM_ROOT_DIR.parent
MASSSPECGYM_DATA_DIR = MASSSPECGYM_REPO_DIR / 'data'
MASSSPECGYM_TEST_RESULTS_DIR = MASSSPECGYM_DATA_DIR / 'test_results'
