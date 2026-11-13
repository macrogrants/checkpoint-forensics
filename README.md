# checkpoint-forensics

This is the code for my report, [Top Chinese Frontier AI Models Carry the Fingerprint of Export-Controlled NVIDIA Blackwell GPUs](https://unemploymentlabs.com/checkpoint-forensics).

Almost every AI model carries a hidden signature of the GPU that created it. You can use this to check that yourself for five models from DeepSeek, Moonshot, MiniMax and Ant Group and it runs on a laptop in a few seconds.

    pip install numpy scipy
    python replay.py

| Model | Seed | SMs | Rows matched | Median r | Seed+1 | Next best | GPU |
|---|---|---|---|---|---|---|---|
| DeepSeek-V3-Base | 953049583 | 132 | 8/8 | 0.999999 | -0.0038 | 183 (0.016) | Hopper |
| DeepSeek-V4-Flash | 953049583 | 148 | 461/469 | 0.999999 | 0.0016 | 209 (0.023) | Blackwell B200/B300 |
| Kimi-K3 | 20256586 | 152 | 13/13 | 0.999996 | 0.0042 | 69 (0.018) | Blackwell GB200/GB300 |
| MiniMax-M3 | 2760 | 148 | 17/17 | 0.999990 | 0.0017 | 87 (0.018) | Blackwell B200/B300 |
| Ling-3.1-flash-base | 3952 | 152 | 384/384 | 0.983436 | 0.0004 | 192 (0.025) | Blackwell GB200/GB300 |

For each model it rebuilds the untouched embedding rows from the seed under every SM count from 16 to 256 and prints the one that matches. seed+1 is a control and next best is the runner-up SM count. V4-Flash is 461/469 because 8 of those rows got partly trained.

## How it works

PyTorch's CUDA `normal_` launches 2,048 threads per SM and every thread gets its own Philox stream. Element `i` is written by thread `i mod G` where `G = 2048 × SMs`. A seed gives the same numbers on any GPU but they end up in different positions depending on the SM count.

| GPU | SMs | Threads (G) |
|---|---|---|
| A100 / A800 | 108 | 221,184 |
| H100 / H800 / H200 | 132 | 270,336 |
| B200 / B300 | 148 | 303,104 |
| GB200 / GB300 | 152 | 311,296 |

Embedding rows for tokens that never appear in training never get a gradient so they still hold the init. With the right seed and SM count you get them back at r > 0.9999. Anything else gives about 0.

The seeds were brute forced over all 2^32 values.

## On a real GPU

    python gpu_check.py

Same check with stock PyTorch (`torch.cuda.manual_seed` then `normal_`) so nothing in this repo is generating the numbers. Rows only match on a GPU with the right SM count. This is what we got on the three GPUs we tried.

| GPU | DeepSeek-V3 | DeepSeek-V4-Flash | Kimi K3 | MiniMax-M3 |
|---|---|---|---|---|
| H100 (132 SMs) | 8/8, bit for bit | 0/469 | | |
| B300 (148 SMs) | 0/8 | 461/469 | 0/13 | 17/17 |
| GB200 (152 SMs) | | | 13/13 | |

Seed+1 matched nothing on any of them.

## Notes

- `rows/` is the rows copied from Hugging Face, 7 MB total. `REFETCH=1 python replay.py` downloads them again from the pinned revisions. They get checked against the hashes in `models.json` either way.
- This is the GPU that created the starting weights. It doesn't say where the training happened after that.
- H100, H800 and H200 are all 132 SMs. GB200 and GB300 are both 152 and so is an AMD MI300X in two-way partition mode. A stock MI300X is 304.
- Ling uses `trunc_normal_` and our model of the bf16 path only gets to r 0.98.
- MiniMax's embedding is split into 4 shards and shard n uses seed 2760 + n.

MIT
