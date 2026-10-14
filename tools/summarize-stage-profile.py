#!/usr/bin/env python3
"""Summarize inclusive eager scopes per GPU and token-count bucket.

Collectives and SSD reads are nested in parent scopes; their durations must not
be added to parent categories. Stream times include submission gaps and waits.
"""
import argparse
import json
from pathlib import Path


def summarize(paths):
    groups = {}
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            tokens = row['tokens'] or 0
            # A shape bucket, not an authoritative scheduler prefill/decode tag.
            bucket = 'tokens_ge_8' if tokens >= 8 else 'tokens_lt_8'
            key = (row['device'], row['kind'], bucket)
            group = groups.setdefault(key, {'device': key[0], 'kind': key[1],
                'shape_bucket': bucket, 'calls': 0, 'token_rows': 0, 'root_stream_ms': 0,
                'root_wall_ms': 0, 'inclusive_scopes': {}})
            group['calls'] += 1
            group['token_rows'] += tokens
            group['root_stream_ms'] += row['root_stream_ms']
            group['root_wall_ms'] += row['root_wall_ms']
            for name, values in row['inclusive_scopes'].items():
                target = group['inclusive_scopes'].setdefault(name,
                    {'calls': 0, 'stream_ms': 0, 'host_dispatch_ms': 0})
                for field in target:
                    target[field] += values[field]
    return {'scope': 'inclusive stream intervals, overlapping across nested categories; do not sum ranks',
            'shape_bucket_warning': 'Token count does not prove scheduler phase; small prefill tails may appear in tokens_lt_8',
            'groups': [groups[key] for key in sorted(groups)]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    paths = sorted(args.directory.glob('stages-*.jsonl'))
    if not paths:
        parser.error('No stage traces found')
    print(json.dumps(summarize(paths), indent=2))
