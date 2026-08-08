import sys

from src.main import main
from src.exceptions import InputFileError


# This file is what makes `python -m src ...` runnable — Python looks for
# __main__.py inside a package when it's invoked with -m. It's a thin
# wrapper around src.main.main(): the try/except here is a second safety
# net around the one already inside main() (see src/main.py, which already
# catches InputFileError around the file-loading step). This outer catch
# covers the case where main() is imported and called some other way and
# an InputFileError still escapes — either way, the process exits cleanly
# with a one-line "[ERROR] ..." message instead of a raw traceback.
if __name__ == "__main__":
    try:
        main()
    except InputFileError as e:
        sys.exit(f"[ERROR] {e}")
