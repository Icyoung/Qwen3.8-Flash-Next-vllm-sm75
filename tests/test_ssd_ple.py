import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from q38ssd.store import RowStore, read_header, read_manifest
from q38ssd.loader import weights_iterator


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_tool("prepare-ssd-ple").prepare
attach = load_tool("attach-ssd-ple").attach

spec = importlib.util.spec_from_file_location("run_ar", ROOT / "quant/run-autoround.py")
ar_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar_runner)


def fixture(directory):
    directory.mkdir()
    # Mixed source shard: PLE and ordinary weights, with rows spanning pages.
    payloads = {
        "model.ple_embedding.ngram_embedding.shard_0.weight": ([19, 160], bytes(range(256)) * 23 + bytes(range(192))),
        "model.ple_embedding.ngram_embedding.shard_1.weight": ([7, 160], bytes(range(224)) * 10),
        "model.layers.0.weight": ([2, 2], b"ordinary"),
    }
    header, content, offset = {}, bytearray(), 0
    for name, (shape, raw) in payloads.items():
        assert len(raw) == shape[0] * shape[1] * 2
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        content.extend(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (directory / "model-00001.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + content)
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {n: "model-00001.safetensors" for n in header}}))
    return payloads


class SSDPLETests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.payloads = fixture(self.root / "source")
        prepare(self.root / "source", self.root / "ple", chunk_bytes=113, expected_shards=2)
        self.manifest = self.root / "ple/ple-manifest.json"
        self.name = "model.ple_embedding.ngram_embedding.shard_0.weight"

    def tearDown(self):
        self.temp.cleanup()

    def test_byte_preserving_extraction_and_mixed_source(self):
        _, manifest = read_manifest(self.manifest)
        self.assertEqual(len(manifest["tensors"]), 2)
        self.assertFalse(any("layers" in name for name in manifest["tensors"]))
        for name, tensor in manifest["tensors"].items():
            path = self.manifest.parent / tensor["file"]
            with path.open("rb") as stream:
                stream.seek(tensor["offset"])
                self.assertEqual(stream.read(), self.payloads[name][1])

    def test_calibration_view_removes_ple_from_headers_and_index(self):
        view = self.root / "view"
        ar_runner.prepare_view(self.root / "source", view)
        mapping = json.loads((view / "model.safetensors.index.json").read_text())["weight_map"]
        self.assertEqual(set(mapping), {"model.layers.0.weight"})
        path = view / mapping["model.layers.0.weight"]
        self.assertFalse(path.is_symlink())
        header, data_start, _ = read_header(path)
        self.assertEqual(set(header), {"model.layers.0.weight"})
        self.assertEqual(path.read_bytes()[data_start:], b"ordinary")

    def test_repeated_out_of_order_and_cross_page_rows(self):
        rows = [18, 0, 12, 12, 13]
        with RowStore(self.manifest, cache_bytes=4096, page_bytes=4096) as reader:
            actual = reader.rows(self.name, rows)
            source = self.payloads[self.name][1]
            self.assertEqual(actual, b"".join(source[r * 320:(r + 1) * 320] for r in rows))
            self.assertLessEqual(reader.cached_bytes, 4096)
            self.assertGreater(reader.hits, 0)

    def test_zero_cache_empty_and_invalid_ids(self):
        with RowStore(self.manifest, cache_bytes=0, page_bytes=127) as reader:
            self.assertEqual(reader.rows(self.name, []), b"")
            self.assertEqual(reader.rows(self.name, [0]), self.payloads[self.name][1][:320])
            self.assertEqual(reader.cached_bytes, 0)
            for rows in [[-1], [19], [1.5], [True]]:
                with self.assertRaises(IndexError):
                    reader.rows(self.name, rows)

    def test_close_and_truncation(self):
        reader = RowStore(self.manifest)
        reader.close()
        with self.assertRaises(RuntimeError):
            reader.rows(self.name, [0])
        tensor = json.loads(self.manifest.read_text())["tensors"][self.name]
        path = self.manifest.parent / tensor["file"]
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            read_manifest(self.manifest)

    def test_header_checksum_and_path_escape(self):
        original = self.manifest.read_text()
        for field, value in [("header_sha256", "0" * 64), ("file", "../escape.safetensors")]:
            manifest = json.loads(original)
            manifest["tensors"][self.name][field] = value
            self.manifest.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                read_manifest(self.manifest)
        self.manifest.write_text(original)

    def test_no_overwrite_and_original_required(self):
        with self.assertRaises(FileExistsError):
            prepare(self.root / "source", self.root / "ple", expected_shards=2)
        with self.assertRaises(ValueError):
            prepare(self.root / "source", self.root / "new-ple", expected_shards=128)
        self.assertFalse((self.root / "new-ple").exists())

    def test_attach_preserves_ordinary_index_and_backup(self):
        model = self.root / "quant"
        model.mkdir()
        index = model / "model.safetensors.index.json"
        original = json.dumps({"metadata": {"total_size": 7}, "weight_map": {"model.layers.0.weight": "quant.safetensors"}})
        index.write_text(original)
        attach(model, self.manifest)
        mapping = json.loads(index.read_text())["weight_map"]
        self.assertEqual(mapping["model.layers.0.weight"], "quant.safetensors")
        self.assertEqual((model / "model.safetensors.index.before-ssd.json").read_text(), original)
        self.assertEqual(len(mapping), 3)
        self.assertTrue((model / mapping[self.name]).is_file())
        self.assertFalse((model / mapping[self.name]).readlink().is_absolute())
        with self.assertRaises(ValueError):
            attach(model, self.manifest)
        # NAS and local SSD use different absolute mount points.
        moved = self.root / "relocated"
        moved.mkdir()
        model.rename(moved / "quant")
        (self.root / "ple").rename(moved / "ple")
        self.assertTrue((moved / "quant" / mapping[self.name]).is_file())

    def test_ordinary_iterator_arguments_are_preserved(self):
        # No torch import or tensor allocation on the non-SSD path.
        from unittest.mock import patch
        seen = []
        def base(*args, **kwargs):
            seen.append((args, kwargs))
            yield "expert", 42
        with patch.dict("os.environ", {"Q38_PLE_SSD": "0"}):
            result = list(weights_iterator(base, ["ordinary"], False, "lazy", {0, 1}, safetensors_prefetch_num_threads=1))
        self.assertEqual(result, [("expert", 42)])
        self.assertEqual(seen, [((["ordinary"], False, "lazy", {0, 1}), {"safetensors_prefetch_num_threads": 1})])

    def test_eager_or_missing_ple_is_rejected_before_loading(self):
        from unittest.mock import patch
        def base(*args, **kwargs):
            self.fail("Ordinary loading started before validation")
            yield
        with patch.dict("os.environ", {"Q38_PLE_SSD": "1", "Q38_PLE_MANIFEST": str(self.manifest)}):
            for strategy in ["eager", "lazy"]:
                with self.assertRaises(ValueError):
                    list(weights_iterator(base, [], False, strategy))

    def test_ssd_iterator_only_requests_meta_ple_and_keeps_ep_filter(self):
        # Lightweight allocation spy: verifies the loader contract, not PyTorch
        # execution or GPU integration. Any attempted CPU PLE allocation fails.
        from types import SimpleNamespace
        from unittest.mock import patch
        calls = []
        def empty(shape, *, dtype, device):
            self.assertEqual(device, "meta")
            self.assertEqual(dtype, "BF16_SPY")
            calls.append(tuple(shape))
            return (tuple(shape), device)
        def base(files, tqdm, strategy, experts, **kwargs):
            self.assertEqual(files, ["ordinary.safetensors"])
            self.assertEqual(experts, {3, 4})
            yield "ordinary", 42
        files = ["ordinary.safetensors"] + [str(p) for p in sorted(self.manifest.parent.glob("*.safetensors"))]
        spy = SimpleNamespace(empty=empty, bfloat16="BF16_SPY")
        with patch.dict("os.environ", {"Q38_PLE_SSD": "1", "Q38_PLE_MANIFEST": str(self.manifest)}), patch.dict(sys.modules, {"torch": spy}):
            result = list(weights_iterator(base, files, False, "lazy", {3, 4}))
        self.assertEqual(result[0], ("ordinary", 42))
        self.assertEqual(set(calls), {(19, 160), (7, 160)})
        self.assertEqual(len(result), 3)

    def test_overlapping_or_duplicate_header_is_rejected(self):
        path = self.root / "bad.safetensors"
        for raw in [b'{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}', b'{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"y":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}']:
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"00")
            with self.assertRaises(ValueError):
                read_header(path)


if __name__ == "__main__":
    unittest.main()
