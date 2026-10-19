#!/usr/bin/env python3
"""Bounded sequential HTTP benchmark; optional eager diagnostic kept separate.

Run against an already monitored server. This tool does not start services,
change power limits, or provide a substitute for an independent thermal guard.
"""
import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

METRICS = ('request_prefill_time_seconds', 'request_decode_time_seconds',
           'request_queue_time_seconds', 'time_to_first_token_seconds')
CASES = [
    ('medium', 'Read these notes and then explain their main idea.\n' +
     'The garden has trees, water, sunlight, birds and flowers. ' * 32),
    ('long', 'Read these notes and then explain their main idea.\n' +
     'A quiet library stores history, science, art and travel books. ' * 64),
]


def metrics(url):
    with urllib.request.urlopen(url + '/metrics', timeout=5) as response:
        text = response.read().decode()
    names = [name + suffix for name in METRICS for suffix in ('_sum', '_count')]
    names += ['spec_decode_num_' + name + '_total' for name in
              ('drafts', 'draft_tokens', 'accepted_tokens')]
    result = {}
    for name in names:
        pattern = re.compile(re.escape('vllm:' + name) + r'(?:\{.*\})? ([^ ]+)(?: .*)?')
        result[name] = sum(float(m[1]) for line in text.splitlines()
                           if (m := pattern.fullmatch(line)))
    return result


def request(url, model, prompt, tokens):
    payload = dict(model=model, messages=[dict(role='user', content=prompt)],
                   temperature=0, seed=42, max_tokens=tokens, ignore_eos=True,
                   stream=True, stream_options={'include_usage': True},
                   chat_template_kwargs={'enable_thinking': False})
    req = urllib.request.Request(url + '/v1/chat/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    first = last = usage = None
    text = ''
    with urllib.request.urlopen(req, timeout=180) as response:
        for line in response:
            if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                continue
            chunk = json.loads(line[6:])
            if chunk.get('error'):
                raise RuntimeError('Server stream error: ' + json.dumps(chunk['error']))
            if chunk.get('usage'):
                usage = chunk['usage']
            for choice in chunk.get('choices', []):
                delta = choice.get('delta', {})
                content = delta.get('content') or delta.get('reasoning_content') or ''
                if content:
                    last = time.monotonic()
                    first = first if first is not None else last
                    text += content
    if not usage or first is None or usage['completion_tokens'] != tokens:
        raise RuntimeError(f'Unexpected completion usage: {usage}')
    return dict(prompt_tokens=usage['prompt_tokens'], output_tokens=tokens,
                ttft_seconds=first-start, total_seconds=time.monotonic()-start,
                output_excerpt=text[:180], output_sha256=hashlib.sha256(text.encode()).hexdigest())


def start_temperature(path):
    if path is None:
        return None
    lines = path.read_text().splitlines()
    if not lines:
        raise RuntimeError('No telemetry sample; wait for the independent monitor')
    row = json.loads(lines[-1])
    if time.time() - row['time'] > 6:
        raise RuntimeError('Stale telemetry')
    if len(row['gpus']) != 4:
        raise RuntimeError('Expected four GPUs')
    temp = max(g['temperature_c'] for g in row['gpus'])
    if temp >= 80:
        raise RuntimeError('Do not start another request at 80C or above')
    return temp


def run(args):
    results = []
    def save():
        args.output.write_text(json.dumps(results, indent=2) + '\n')
    for name, prompt in [('hello', 'Say hello in one sentence.')] + CASES:
        temp = start_temperature(args.telemetry)
        row = request(args.url, args.model, prompt, 8)
        row.update(case=name, phase='warmup', temperature_start_max_c=temp)
        results.append(row)
        save()
        time.sleep(2)
    for repeat in range(args.repeats):
        for name, prompt in CASES if repeat % 2 == 0 else reversed(CASES):
            temp = start_temperature(args.telemetry)
            before = metrics(args.url)
            row = request(args.url, args.model, prompt, args.tokens)
            time.sleep(2)
            after = metrics(args.url)
            for metric in METRICS:
                if after[metric+'_count'] - before[metric+'_count'] != 1:
                    raise RuntimeError('Concurrent traffic or missing metrics: ' + metric)
                row['server_'+metric] = after[metric+'_sum'] - before[metric+'_sum']
            row.update(case=name, phase='measured', repeat=repeat,
                       temperature_start_max_c=temp)
            row['prefill_tps'] = row['prompt_tokens'] / row['server_request_prefill_time_seconds']
            row['decode_tps'] = (args.tokens-1) / row['server_request_decode_time_seconds']
            row['speculation'] = {k: after[k]-before[k] for k in before if k.startswith('spec_')}
            if args.require_mtp and row['speculation']['spec_decode_num_drafts_total'] <= 0:
                raise RuntimeError('No MTP drafts observed')
            results.append(row)
            save()
            print(json.dumps(row), flush=True)
    if args.profile_control:
        start_temperature(args.telemetry)
        # The worker reads the completed JSON at the next step boundary.
        pending = args.profile_control.with_suffix('.pending')
        pending.write_text(json.dumps({'enabled': True, 'tag': args.output.stem}))
        pending.replace(args.profile_control)
        try:
            row = request(args.url, args.model, CASES[1][1], args.tokens)
            args.output.with_suffix('.diagnostic.json').write_text(json.dumps(row, indent=2) + '\n')
        finally:
            args.profile_control.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18440')
    parser.add_argument('--model', default='Qwen3.8-Flash-Next-trial')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--telemetry', type=Path)
    parser.add_argument('--profile-control', type=Path)
    parser.add_argument('--require-mtp', action='store_true')
    args = parser.parse_args()
    if args.tokens < 2 or args.repeats < 1:
        parser.error('tokens >= 2 and repeats >= 1 are required')
    if args.output.exists() or (args.profile_control and args.profile_control.exists()):
        parser.error('Use a new output path and remove any existing profile control first')
    run(args)
