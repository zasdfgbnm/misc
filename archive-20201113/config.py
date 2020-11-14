import torch
print(*torch.__config__.show().split("\n"), sep="\n")

