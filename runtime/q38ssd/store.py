"""Read safetensors headers and requested BF16 rows without loading the table.

The explicit LRU is bounded. OS page cache is separate, reclaimable, and must
be accounted for by the container memory limit when measuring host RAM.
"""
import hashlib
import json
import math
import os
import re
import struct
from collections import OrderedDict
from pathlib import Path

PLE_NAME = re.compile(r"^(.*\.ngram_embedding)\.shard_(\d+)\.weight$")
MAX_HEADER_BYTES = 16 * 1024 * 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_header(path):
    """Validate a safetensors header; do not mmap or read tensor contents."""
    path = Path(path)
    with path.open("rb") as stream:
        first = stream.read(8)
        if len(first) != 8:
            raise ValueError(f"Truncated safetensors header: {path}")
        size = struct.unpack("<Q", first)[0]
        if not 2 <= size <= MAX_HEADER_BYTES:
            raise ValueError(f"Invalid safetensors header size: {path}")
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError(f"Truncated safetensors header: {path}")
    header = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise ValueError("Header must be a JSON object")
    data_size = path.stat().st_size - 8 - size
    ranges = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        shape, offsets = tensor["shape"], tensor["data_offsets"]
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"Invalid tensor shape: {name}")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets):
            raise ValueError(f"Invalid tensor offsets: {name}")
        start, end = offsets
        if not 0 <= start <= end <= data_size:
            raise ValueError(f"Tensor outside file: {name}")
        if tensor["dtype"] == "BF16" and end - start != math.prod(shape) * 2:
            raise ValueError(f"Incorrect BF16 tensor size: {name}")
        ranges.append((start, end))
    ranges.sort()
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError("Overlapping tensor offsets")
    return header, 8 + size, hashlib.sha256(first + raw).hexdigest()


def read_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    if manifest.get("schema") != 1 or manifest.get("dtype") != "BF16":
        raise ValueError("Expected schema 1 BF16 PLE manifest")
    tensors = manifest["tensors"]
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("Empty PLE manifest")
    files = {}
    for name, record in tensors.items():
        if not PLE_NAME.fullmatch(name):
            raise ValueError(f"Unexpected PLE tensor name: {name}")
        filename = record["file"]
        if Path(filename).name != filename or not filename.startswith("q38-ple-"):
            raise ValueError("PLE files must be dedicated files in the manifest directory")
        resolved = (path.parent / filename).resolve()
        if resolved.parent != path.parent:
            raise ValueError("PLE store files cannot escape their directory")
        if filename not in files:
            header, data_start, digest = read_header(resolved)
            if digest != record["header_sha256"]:
                raise ValueError(f"PLE header checksum mismatch: {filename}")
            if any(n != "__metadata__" and not PLE_NAME.fullmatch(n) for n in header):
                raise ValueError("Dedicated PLE shard contains ordinary model weights")
            files[filename] = (header, data_start, resolved)
        header, data_start, _ = files[filename]
        entry = header[name]
        if entry["dtype"] != "BF16" or len(entry["shape"]) != 2:
            raise ValueError("Only two-dimensional BF16 PLE tensors are supported")
        if entry["shape"] != record["shape"] or data_start + entry["data_offsets"][0] != record["offset"]:
            raise ValueError(f"PLE geometry/offset mismatch: {name}")
        if record["header_sha256"] != read_header(path.parent / filename)[2]:
            raise ValueError(f"PLE header checksum mismatch: {filename}")
    actual_names = {n for h, _, _ in files.values() for n in h if n != "__metadata__"}
    if actual_names != set(tensors):
        raise ValueError("Manifest does not cover every PLE tensor in its files")
    return path, manifest


class RowStore:
    """Single-worker reader. Repeated rows/pages share a bounded LRU cache."""

    def __init__(self, manifest_path, cache_bytes=32 * 1024 * 1024, page_bytes=4096):
        if type(cache_bytes) is not int or cache_bytes < 0 or type(page_bytes) is not int or page_bytes <= 0:
            raise ValueError("Invalid cache/page size")
        self.path, self.manifest = read_manifest(manifest_path)
        self.tensors = self.manifest["tensors"]
        self.page_bytes = page_bytes
        self.capacity = cache_bytes // page_bytes
        self.cache = OrderedDict()
        self.fds = {}
        self.disk_bytes = 0
        self.hits = 0
        self.closed = False

    @property
    def cached_bytes(self):
        return sum(len(value) for value in self.cache.values())

    def _page(self, filename, page):
        if self.closed:
            raise RuntimeError("PLE reader is closed")
        key = (filename, page)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        if filename not in self.fds:
            self.fds[filename] = os.open(self.path.parent / filename, os.O_RDONLY)
        fd = self.fds[filename]
        offset = page * self.page_bytes
        expected = min(self.page_bytes, os.fstat(fd).st_size - offset)
        if expected <= 0:
            raise OSError("PLE read outside file")
        pieces = []
        count = 0
        while count < expected:
            piece = os.pread(fd, expected - count, offset + count)
            if not piece:
                raise OSError("Short PLE read")
            pieces.append(piece)
            count += len(piece)
        data = b"".join(pieces)
        self.disk_bytes += len(data)
        if self.capacity:
            self.cache[key] = data
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return data

    def rows(self, name, row_ids):
        tensor = self.tensors[name]
        num_rows, width = tensor["shape"]
        row_bytes = width * 2
        result = bytearray()
        for row in row_ids:
            if type(row) is not int or not 0 <= row < num_rows:
                raise IndexError(f"PLE row out of range: {row}")
            offset = tensor["offset"] + row * row_bytes
            remaining = row_bytes
            while remaining:
                page, within = divmod(offset, self.page_bytes)
                data = self._page(tensor["file"], page)
                size = min(remaining, len(data) - within)
                if size <= 0:
                    raise OSError("Truncated PLE row")
                result.extend(data[within:within + size])
                offset += size
                remaining -= size
        return result

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()
        self.cache.clear()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
