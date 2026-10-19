import importlib.util
import io
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'native_benchmark', Path(__file__).resolve().parents[1] / 'tools/benchmark-native.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class NativeBenchmarkTest(unittest.TestCase):
    def test_mtp_stream_events_are_not_used_as_token_count(self):
        stream = io.BytesIO(
            b'data: {"choices":[{"delta":{"content":"several tokens"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":" at once"}}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":439,"completion_tokens":64}}\n\n'
            b'data: [DONE]\n\n')
        with patch.object(benchmark.urllib.request, 'urlopen', return_value=stream):
            row = benchmark.request('http://localhost', 'test', 'prompt', 64)
        self.assertEqual(row['output_tokens'], 64)
        self.assertEqual(row['prompt_tokens'], 439)
        self.assertEqual(row['output_excerpt'], 'several tokens at once')

    def test_stream_error_is_not_reported_as_success(self):
        stream = io.BytesIO(b'data: {"error":{"message":"engine failed"}}\n\n')
        with patch.object(benchmark.urllib.request, 'urlopen', return_value=stream):
            with self.assertRaisesRegex(RuntimeError, 'Server stream error.*engine failed'):
                benchmark.request('http://localhost', 'test', 'prompt', 64)

    def test_truncated_completion_is_rejected(self):
        stream = io.BytesIO(
            b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            b'data: {"usage":{"prompt_tokens":439,"completion_tokens":3}}\n\n')
        with patch.object(benchmark.urllib.request, 'urlopen', return_value=stream):
            with self.assertRaisesRegex(RuntimeError, 'Unexpected completion usage'):
                benchmark.request('http://localhost', 'test', 'prompt', 64)


if __name__ == '__main__':
    unittest.main()
