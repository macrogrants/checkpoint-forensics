# rows/ has each model's rows already. REFETCH=1 pulls them again from the pinned Hugging Face revision
# with range requests. either way they have to match the sha256 in models.json
import hashlib, json, os, struct, urllib.request
import numpy as np


def get(url, a=None, b=None):
    req = urllib.request.Request(url, headers={"User-Agent": "checkpoint-forensics"})
    if a is not None:
        req.add_header("Range", f"bytes={a}-{b}")
    return urllib.request.urlopen(req, timeout=120).read()


def rows(m):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rows", m["name"] + ".npy")
    if os.path.exists(path) and not os.environ.get("REFETCH"):
        bits = np.load(path)
    else:
        base = f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/"
        shard = base + json.loads(get(base + "model.safetensors.index.json"))["weight_map"][m["tensor"]]
        n = struct.unpack("<Q", get(shard, 0, 7))[0]
        info = json.loads(get(shard, 8, 7 + n))[m["tensor"]]
        assert info["dtype"] == "BF16" and info["shape"] == m["shape"]
        d, at = m["shape"][1], 8 + n + info["data_offsets"][0]
        bits = np.stack([np.frombuffer(get(shard, at + r * d * 2, at + (r + 1) * d * 2 - 1), "<u2") for r in m["rows"]])
        np.save(path, bits)
    assert hashlib.sha256(bits.astype("<u2").tobytes()).hexdigest() == m["sha256"], m["name"] + " rows don't match the hash"
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)  # bf16 bits -> float
