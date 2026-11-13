# same check with stock PyTorch on whatever GPU you have: seed it, call normal_, compare.
# a model's rows only match if this GPU has the SM count that model was initialized on
import json, sys
import numpy as np, torch
from fetch import rows


def gen(m, shift):
    V, d = m["shape"]
    S = -(-V // m.get("tp", 1))
    out = np.zeros((len(m["rows"]), d))
    for rank in sorted({r // S for r in m["rows"]}):
        torch.cuda.manual_seed(m["seed"] + shift + rank * m.get("seed_per_rank", 0))
        t = torch.empty((min(S, V - rank * S), d), dtype=getattr(torch, m["dtype"]), device="cuda").normal_(0, m.get("std", 1))
        for i, r in enumerate(m["rows"]):
            if r // S == rank:
                out[i] = t[r % S].float().cpu().numpy()
        del t
    return out


def corr(a, b):
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


p = torch.cuda.get_device_properties(0)
print(f"{p.name}, {p.multi_processor_count} SMs, torch {torch.__version__}")
print(f"{'model':20}{'init on':>9}{'rows':>10}{'median r':>10}{'seed+1':>10}")
for m in json.load(open("models.json")):
    if (sys.argv[1:] and m["name"] not in sys.argv[1:]) or m["dist"] != "normal":
        continue
    Y = rows(m)
    r, ctl = corr(gen(m, 0), Y), corr(gen(m, 1), Y)
    print(f"{m['name']:20}{m['sms']:>5} SMs{f'{(r > 0.99).sum()}/{len(r)}':>10}{np.median(r):>10.6f}{f'{(ctl > 0.99).sum()}/{len(r)}':>10}")
