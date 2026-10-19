import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    spec = importlib.util.spec_from_file_location('subject', ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StageProfileTest(unittest.TestCase):
    def test_scheduler_phase_handles_tails_and_mixed_batches(self):
        classify = load('runtime/q38ssd/stage_profile.py').scheduler_phase
        self.assertEqual(classify([854], [855]), 'prefill')
        self.assertEqual(classify([855], [855]), 'decode')
        self.assertEqual(classify([859], [855]), 'decode')
        self.assertEqual(classify([854, 859], [855, 855]), 'mixed')
        self.assertEqual(classify([], []), 'empty')

    def test_summary_separates_tags_phases_and_merges_payload_histograms(self):
        summarize = load('tools/summarize-stage-profile.py').summarize
        base = dict(device=0, kind='runner_execute', tokens=1,
                    root_stream_ms=2, root_wall_ms=3,
                    inclusive_scopes={'collective_all_reduce': {
                        'calls': 2, 'stream_ms': 1, 'host_dispatch_ms': .5,
                        'input_bytes_histogram': {'5120': 2}}})
        rows = [dict(base, tag='a', phase='prefill'),
                dict(base, tag='a', phase='decode'),
                dict(base, tag='a', phase='decode'),
                dict(base, tag='b', phase='decode'),
                dict(base, kind='target')]  # legacy trace without tag/phase
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stages-test.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            groups = summarize([path])['groups']
        self.assertEqual(len(groups), 4)
        group = next(g for g in groups if g['tag'] == 'a' and g['phase'] == 'decode')
        self.assertEqual(group['calls'], 2)
        self.assertEqual(group['inclusive_scopes']['collective_all_reduce']['input_bytes_histogram'], {'5120': 4})


if __name__ == '__main__':
    unittest.main()
