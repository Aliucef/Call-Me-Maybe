import sys

from src.main import main
from src.exceptions import InputFileError


if __name__ == "__main__":
    try:
        main()
    except InputFileError as e:
        sys.exit(f"[ERROR] {e}")
