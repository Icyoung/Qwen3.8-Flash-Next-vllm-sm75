import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from q38ssd.hc_small_batch import supported_layout


class HCSmallBatchTests(unittest.TestCase):
    def test_sm75_small_shapes_only(self):
        for rows in (1, 2, 3, 4):
            self.assertTrue(supported_layout(rows, 2560, 320, 4, 'torch.float16', True, (7, 5)))
        for rows in (0, 5, 128):
            self.assertFalse(supported_layout(rows, 2560, 320, 4, 'torch.float16', True, (7, 5)))

    def test_other_layouts_fall_back(self):
        valid = [4, 2560, 320, 4, 'torch.float16', True, (7, 5)]
        for index, value in enumerate([4, 4096, 256, 2, 'torch.bfloat16', False, (8, 0)]):
            if index == 0:
                continue
            args = valid.copy()
            args[index] = value
            self.assertFalse(supported_layout(*args))


if __name__ == '__main__':
    unittest.main()
