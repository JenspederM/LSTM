# %%
import numpy as np

x = np.zeros(100)
x1 = np.zeros((100, 50))
r = np.random.random(50)
s = np.hstack((r, x))
print(x.shape)
print(x1.shape)
print(r.shape)
print(s.shape)

# %%
print(s)
dd = np.outer(x, s)
print(dd.shape)

wg = np.random.rand(100, 50) * (0.1 - -0.1) + -0.1
