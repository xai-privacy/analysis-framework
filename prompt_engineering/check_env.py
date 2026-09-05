#!/usr/bin/env python3
"""Environment diagnostic. Run this first when imports fail.

    python check_env.py

Reports which interpreter is actually running, whether a virtualenv is active,
and which required packages are importable. Uses only the standard library, so
it runs even when nothing else is installed.
"""

import importlib
import sys
from pathlib import Path

REQUIRED = [
    ("requests", "requests", "Ollama HTTP calls"),
]

def main():
    print("=" * 62)
    print("INTERPRETER")
    print("=" * 62)
    print(f"  executable : {sys.executable}")
    print(f"  version    : {sys.version.split()[0]}")

    in_venv = sys.prefix != sys.base_prefix
    print(f"  venv active: {in_venv}")
    if in_venv:
        print(f"  venv path  : {sys.prefix}")

    major, minor = sys.version_info[:2]
    ok_version = (major, minor) >= (3, 9)
    if not ok_version:
        print(f"\n  !! Python {major}.{minor} is older than 3.9.")

    print()
    print("=" * 62)
    print("PACKAGES")
    print("=" * 62)
    missing = []
    for mod, dist, purpose in REQUIRED:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "")
            loc = Path(getattr(m, "__file__", "") or "").parent
            print(f"  [ok]      {mod:<10} {ver:<10} ({purpose})")
            if in_venv and str(sys.prefix) not in str(loc):
                print(f"            ^ warning: loaded from outside the venv: {loc}")
        except ImportError as e:
            missing.append((mod, dist))
            print(f"  [MISSING] {mod:<10} {'':<10} ({purpose})  -> pip install {dist}")
            # Distinguish "not installed" from "installed but fails to import",
            # which are different problems with different fixes.
            if "No module named" not in str(e):
                print(f"            ^ present but failed to import: {e}")

    print()
    print("=" * 62)
    if not missing and ok_version:
        print("All good. Try:  python run_01_zero_shot.py --dry-run --limit 3")
        return 0

    print("DIAGNOSIS")
    print("=" * 62)
    if missing and not in_venv:
        print("  No virtualenv is active, and required packages are missing from")
        print("  this interpreter. You most likely installed into a venv but are")
        print("  running the system Python. Activate it first:")
        print()
        print("      .\\.venv\\Scripts\\Activate.ps1        # PowerShell")
        print()
        print("  Your prompt should then start with (.venv). If activation is")
        print("  blocked by execution policy, either run:")
        print()
        print("      Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass")
        print()
        print("  or skip activation and call the venv's python directly:")
        print()
        print("      .\\.venv\\Scripts\\python.exe run_framework.py --dry-run --n 2")
    elif missing and in_venv:
        print("  A venv is active but packages are missing. Install them into it:")
        print()
        print("      python -m pip install -r requirements.txt")
    elif not ok_version:
        print("  Upgrade to Python 3.9 or newer.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
