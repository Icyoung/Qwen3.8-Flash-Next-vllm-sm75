"""Opt-in INC bridge, small-batch HC fusion and stage scopes for workers.

Put this runtime directory on PYTHONPATH. INC requires Q38_INC_INT3=1; stage
scopes require Q38_STAGE_PROFILE_CONTROL; HC requires Q38_HC_SMALL_BATCH=1.
Unsupported dependencies fail closed.
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

if os.environ.get('Q38_HC_SMALL_BATCH') == '1':
    try:
        from q38ssd.hc_small_batch import install
        install()
    except Exception:
        import traceback
        traceback.print_exc()
        os._exit(78)

if os.environ.get('Q38_STAGE_PROFILE_CONTROL'):
    try:
        from q38ssd.stage_profile import install
        install()
    except Exception:
        import traceback
        traceback.print_exc()
        os._exit(78)
