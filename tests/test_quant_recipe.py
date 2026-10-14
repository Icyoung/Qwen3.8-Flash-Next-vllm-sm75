import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quant"))
from runtime_recipe import linear_scheme


class QuantRecipeTests(unittest.TestCase):
    def test_every_expert_layer_has_requested_bits(self):
        for layer in range(48):
            for projection, n, k in [("gate", 640, 2560), ("up", 640, 2560), ("down", 2560, 640)]:
                name = f"model.language_model.layers.{layer}.mlp.experts.511.{projection}_proj"
                scheme = linear_scheme(name, k, n)
                self.assertEqual(scheme["bits"], 4 if layer in [0, 1, 46, 47] else 3)
                self.assertEqual(scheme["group_size"], 128)

    def test_control_parameters_preserved(self):
        for suffix, n, k in [("mlp.gate", 512, 2560), ("mlp.shared_expert_gate", 1, 2560), ("attn_hyper_connection.input_mix_weight_up", 10240, 320)]:
            self.assertEqual(linear_scheme("model.language_model.layers.2." + suffix, k, n)["bits"], 16)

    def test_small_gdn_projection_explicitly_preserved(self):
        self.assertEqual(linear_scheme("model.language_model.layers.2.linear_attn.in_proj_a", 2560, 48)["bits"], 16)

    def test_tp4_shared_down_uses_compatible_group(self):
        scheme = linear_scheme("model.language_model.layers.2.mlp.shared_expert.down_proj", 640, 2560)
        self.assertEqual((scheme["bits"], scheme["group_size"]), (4, 32))
        self.assertEqual((640 // 4) % scheme["group_size"], 0)


if __name__ == "__main__":
    unittest.main()
