import torch as th

N = 2 ** 25
#N = 16909321 # << this one won't fail

K = 2 ** 7

print(f"Allocating {2*N*K / (1024**3)} GB ...")
U = th.zeros((N, K), dtype=th.half, device="cuda:0")
print("Done.")

u = U[:, 0:1]
#u = th.zeros_like(u) # << adding this makes the following line succeed for either value of N

print(u.shape, u.stride())

ut = th.zeros((1, N), dtype=th.half, device="cuda:0")
v = th.mm(ut, u)
#v = (u**2).sum() # << this succeeds for either value of N

# Just to force the error to propagate up to Python, otherwise it will silently
# fail and exit.
th.cuda.synchronize()
