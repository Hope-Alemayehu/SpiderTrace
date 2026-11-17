MWPM: 664 errors on 200000 big-test shots.

| Config | Seed | Big-test errors | Ratio to MWPM | Fixed vs C0 | Newly wrong vs C0 | McNemar p | Zero-fired predicted flips | Epochs (best) |
|---|---|---|---|---|---|---|---|---|
| C0 | 0 | 1348 | 2.030 | 0 | 0 | 1 | 0/15307 | 69 (59) |
| C0 | 1 | 1429 | 2.152 | 0 | 0 | 1 | 0/15307 | 71 (61) |
| C0 | pooled | | | 0 | 0 | 1 | | |
| C2 | 0 | 1477 | 2.224 | 321 | 450 | 3.84e-06 | 0/15307 | 48 (38) |
| C2 | 1 | 1379 | 2.077 | 444 | 394 | 0.0905 | 0/15307 | 92 (82) |
| C2 | pooled | | | 765 | 844 | 0.0518 | | |

Beats C0 at every seed (fewer errors and p < 0.05): none
