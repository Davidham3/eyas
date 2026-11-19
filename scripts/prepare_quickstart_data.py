#!/usr/bin/env python3
"""Compatibility shim for the historical quickstart data-prep script path.

The implementation now ships inside the package as ``eyas.prepare`` (so the
capability survives a ``pip install eyas`` with no repo checkout); the CLI route
is ``eyas prepare --dataset ...``. This shim keeps the old
``python scripts/prepare_quickstart_data.py`` invocation working for dev flows.
"""

from eyas.prepare import main

if __name__ == "__main__":
    import sys

    sys.exit(main())
