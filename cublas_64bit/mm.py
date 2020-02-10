import torch as th

N = 16909322
#N = 16909321 # << this one won't fail

K = 127

print(f"Allocating {4*N*K / (1024**3)} GB ...")
U = th.zeros((N, K), device="cuda:0")
print("Done.")

u = U[:, 0:1]
#u = th.zeros_like(u) # << adding this makes the following line succeed for either value of N

v = th.mm(u.permute(1, 0), u)
#v = (u**2).sum() # << this succeeds for either value of N

# Just to force the error to propagate up to Python, otherwise it will silently
# fail and exit.
th.cuda.synchronize()
