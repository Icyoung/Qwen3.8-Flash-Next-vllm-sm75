import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from q38ssd.inc_int3 import expert_scheme


class ExpertSchemeTests(unittest.TestCase):
    def recipe(self, bits=3):
        return {f'model.layers.2.mlp.experts.{i}.{p}_proj': {'bits': bits, 'group_size': 128}
                for i in range(512) for p in ('gate', 'up', 'down')}

    def test_middle_and_edge_expert_defaults(self):
        self.assertEqual(expert_scheme(self.recipe(), 'model.layers.2.mlp.experts', 4, 32), (3, 128, True))
        r = self.recipe(4)
        for value in r.values():
            value.pop('bits')
        self.assertEqual(expert_scheme(r, 'model.layers.2.mlp.experts', 4, 32), (4, 128, True))

    def test_missing_and_mixed_metadata_rejected(self):
        r = self.recipe()
        r.pop(next(iter(r)))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            expert_scheme(r, 'model.layers.2.mlp.experts', 4, 32)
        r = self.recipe()
        r[next(iter(r))]['bits'] = 4
        with self.assertRaisesRegex(ValueError, 'Mixed'):
            expert_scheme(r, 'model.layers.2.mlp.experts', 4, 32)


if __name__ == '__main__':
    unittest.main()
