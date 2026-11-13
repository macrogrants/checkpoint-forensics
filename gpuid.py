# CPU model of PyTorch's CUDA normal_ (curand Philox4x32-10), aten/src/ATen/native/cuda/DistributionTemplates.h
# G = 2048 * SMs threads. thread t uses Philox subsequence t and each call writes 4 values G apart:
#   t = i % G    lane = (i // G) % 4    call = i // 4G    counter = (offset/4 + call, 0, t, 0)    key = seed
# the first G elements come out the same for every G, so only positions past that can tell layouts apart
import numpy as np

M0, M1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
W0, W1 = 0x9E3779B9, 0xBB67AE85
LO, S = np.uint64(0xFFFFFFFF), np.uint64(32)

KAT = [  # Random123 test vectors
    ((0x00000000, 0x00000000, 0x00000000, 0x00000000), (0x00000000, 0x00000000),
     (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
    ((0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), (0xffffffff, 0xffffffff),
     (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
    ((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0),
     (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1)),
]


def philox(c0, c1, c2, c3, k0, k1):
    c0, c1, c2, c3 = (np.asarray(x, np.uint64) & LO for x in (c0, c1, c2, c3))
    k0, k1 = int(k0) & 0xFFFFFFFF, int(k1) & 0xFFFFFFFF
    for _ in range(10):
        p0, p1 = M0 * c0, M1 * c2
        c0, c1, c2, c3 = (p1 >> S) ^ c1 ^ np.uint64(k0), p1 & LO, (p0 >> S) ^ c3 ^ np.uint64(k1), p0 & LO
        k0, k1 = (k0 + W0) & 0xFFFFFFFF, (k1 + W1) & 0xFFFFFFFF
    return c0, c1, c2, c3


def draw(pos, G, seed, offset=0, dist="normal"):
    p, G = np.asarray(pos, np.uint64), np.uint64(G)
    t, lane = p % G, ((p // G) % np.uint64(4)).astype(np.int64)
    call = np.uint64(offset // 4) + p // (np.uint64(4) * G)
    w = np.stack(philox(call & LO, call >> S, t & LO, t >> S, seed, seed >> 32)).astype(np.float64)
    i, inv = np.arange(p.size), np.float32(2.0 ** -32)
    if dist == "trunc_normal":  # uniform_ then erfinv, with the bounds as bf16 rounds them
        from scipy.special import erfinv
        u = w[lane, i].astype(np.float32) * inv + inv / np.float32(2)
        return erfinv(u.astype(np.float64) * 1.9921875 - 0.99609375)
    # curand_normal4: Box-Muller on words (0,1) and (2,3), sin on even lanes and cos on odd
    x = w[np.where(lane < 2, 0, 2), i].astype(np.float32)
    y = w[np.where(lane < 2, 1, 3), i].astype(np.float32)
    u = x * inv + inv / np.float32(2)
    tp = inv * np.float32(6.2831855)
    v = y * tp + tp / np.float32(2)
    s = np.sqrt(np.float32(-2) * np.log(u))
    return np.where(lane % 2 == 0, np.sin(v) * s, np.cos(v) * s).astype(np.float64)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


for c, k, want in KAT:
    assert tuple(map(int, philox(*c, *k))) == want
