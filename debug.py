import numpy as np


k1 = np.array([2.0])
k2 = np.array([2])
m = np.array([2])

n1 = m.astype(k1.dtype, copy=False)
n2 = m.astype(k2.dtype, copy=True)
pass

print(np.newaxis)