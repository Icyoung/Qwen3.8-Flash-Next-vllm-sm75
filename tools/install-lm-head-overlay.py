#!/usr/bin/env python3
"""Pass INC quantization to the pinned Qwen4Exp main or MTP LM head.

Preserves the checkpoint's W4 head; no dequantization or weight rewriting.
Only the reviewed NVIDIA sources are accepted; --mtp-only selects MTP.
"""
import argparse
import ast
import hashlib
from pathlib import Path

RELATIVE = 'models/qwen4_exp/nvidia/model.py'
EXPECTED = '4f80224ea138e7b130f2d2a2bccec049a9deb98de2e00a681ad23d7502f97a26'
OLD = '''        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )'''
NEW = OLD.replace('            prefix=', '            quant_config=self.quant_config,\n            prefix=')
MTP_RELATIVE = 'models/qwen4_exp/nvidia/mtp.py'
MTP_EXPECTED = '91f824ac4205e295ea48741ead8386a67a0e1d5d2eb9db0777f57afd0a0f0699'
MTP_OLD = '''                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),'''
MTP_NEW = MTP_OLD.replace('                    prefix=',
                          '                    quant_config=self.quant_config,\n                    prefix=')


def patch_source(data, mtp=False):
    expected, old, new = (MTP_EXPECTED, MTP_OLD, MTP_NEW) if mtp else (EXPECTED, OLD, NEW)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Unreviewed Qwen4Exp source; rebuild from the pinned base')
    source = data.decode()
    if source.count(old) != 1:
        raise ValueError('Expected exactly one LM head constructor')
    result = source.replace(old, new, 1)
    ast.parse(result)
    return result.encode()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--mtp-only', action='store_true', help='Patch the standalone MTP head instead of the main model')
    args = parser.parse_args()
    relative = MTP_RELATIVE if args.mtp_only else RELATIVE
    path = args.package / relative
    result = patch_source(path.read_bytes(), mtp=args.mtp_only)
    path.write_bytes(result)
    print(relative, hashlib.sha256(result).hexdigest())
