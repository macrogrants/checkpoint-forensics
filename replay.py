# regenerate each model's untouched embedding rows on CPU and see which GPU layout they match
import json, sys
import numpy as np
from gpuid import draw, corr
from fetch import rows

GPU = {132: "Hopper", 148: "Blackwell B200/B300", 152: "Blackwell GB200/GB300"}


def match(m, Y, sms, shift=0):
    V, d = m["shape"]
    S = -(-V // m.get("tp", 1))  # rows per tensor-parallel shard, each shard has its own seed
    out = []
    for r, y in zip(m["rows"], Y):
        pos = np.uint64(r % S * d) + np.arange(d, dtype=np.uint64)
        seed = m["seed"] + shift + r // S * m.get("seed_per_rank", 0)
        out.append(corr(y, draw(pos, sms * 2048, seed, m.get("cpu_offset", 0), m["dist"])))
    return np.array(out)


print(f"{'model':20}{'seed':>10}{'SMs':>6}{'rows':>10}{'median r':>10}{'seed+1':>9}{'next best':>14}  gpu")
for m in json.load(open("models.json")):
    if sys.argv[1:] and m["name"] not in sys.argv[1:]:
        continue
    Y = rows(m)
    few = dict(m, rows=m["rows"][:4])
    sweep = {s: np.median(match(few, Y[:4], s)) for s in range(16, 257)}
    best, nxt = sorted(sweep, key=sweep.get, reverse=True)[:2]
    r, ctl = match(m, Y, best), match(m, Y, best, 1)
    hit = f"{(r > m.get('thr', 0.99)).sum()}/{len(r)}"
    print(f"{m['name']:20}{m['seed']:>10}{best:>6}{hit:>10}{np.median(r):>10.6f}{np.median(ctl):>9.4f}"
          f"{f'{nxt} ({sweep[nxt]:.3f})':>14}  {GPU.get(best, '?')}")
