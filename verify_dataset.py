import numpy as np
import glob

files = sorted(glob.glob('data/*.npz'))
print(f'{len(files)} files found\n')

all_pass = True
for f in files:
    d = np.load(f)
    n = len(d['syndrome_bits'])
    nz_zx = (d['zx_features'].sum(axis=1) > 0).sum()
    nz_syn = (d['syndrome_bits'].sum(axis=1) > 0).sum()
    flips = d['logical_flip'].sum()
    
    # extract p from filename for pass criteria
    p = float(f.split('_p')[1].replace('.npz', ''))
    
    flip_ok = flips > 0 if p >= 0.01 else True
    syn_ok = nz_syn > 0
    zx_ok = nz_zx > 0
    status = 'PASS' if (flip_ok and syn_ok and zx_ok) else 'FAIL'
    if status == 'FAIL':
        all_pass = False
    
    print(f'{status} | {f} | shots={n} | syn_nonzero={nz_syn} | zx_nonzero={nz_zx} | flips={flips}')

print(f'\n{"ALL PASS" if all_pass else "SOME FAILED"}')