import numpy as np
import pathlib
import sys
p = pathlib.Path('service/embeddings_cache.npz')
if not p.exists():
    print('No cache file found at', p)
    sys.exit(0)
data = np.load(p, allow_pickle=True)
cache = data['cache'].item()
print('Cache keys:')
for k in sorted(cache.keys()):
    v = cache[k]
    labels = v.get('labels', [])
    print(f'  {k}: {len(labels)} sentences; unique labels: {sorted(set(labels))}')
