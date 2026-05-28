import numpy as np

data = np.load("data/d3_p0.0100.npz")

print(list(data.keys()))                    # ['syndrome_bits', 'zx_features', 'logical_flip']
print(data["syndrome_bits"].shape)          # (1000, 8)
print(data["zx_features"].shape)            # (1000, 9)
print(data["logical_flip"].shape)           # (1000,)

# Quick stats
print((data["zx_features"] > 0).any(axis=1).sum(), "shots with non-zero ZX features")
print(data["logical_flip"].sum(), "logical flips")
