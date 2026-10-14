"""Opt-in INC bridge for the API process and every spawned worker.

Put this runtime directory on PYTHONPATH. No imports or changes occur unless
Q38_INC_INT3=1; unsupported pinned dependencies fail before serving begins.
"""
import os

if os.environ.get('Q38_INC_INT3') == '1':
    try:
        from q38ssd.inc_int3 import install
        install()
    except Exception:
        import traceback
        traceback.print_exc()
        os._exit(78)
