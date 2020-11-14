import torch
from itertools import product

device = 'cuda'
dtype = torch.float64

def assertEqual(a, b):
    print((a - b).abs().amax())

def run_test(tensor_dims, some):
    A = torch.randn(*tensor_dims, device=device, dtype=dtype)
    Q, R = torch.qr(A, some=some)

    # Check0: Q[-2:] = (m, n_columns), R[-2:] = (n_columns, n)
    m, n = tensor_dims[-2:]
    n_columns = m if (not some) and m > n else min(m, n)

    # Check1: A = QR
    assertEqual(A, torch.matmul(Q, R))

    # Check4: Q^{T}Q = I
    assertEqual(torch.matmul(Q.transpose(-2, -1), Q),
                        torch.eye(n_columns, device=device).expand(Q.shape[:-2] + (n_columns, n_columns)))

tensor_dims_list = [(3, 5), (5, 5), (5, 3),  # Single matrix
                    (7, 3, 5), (7, 5, 5), (7, 5, 3),  # 3-dim Tensors
                    (7, 5, 3, 5), (7, 5, 5, 5), (7, 5, 5, 3)]  # 4-dim Tensors

for tensor_dims, some in product(tensor_dims_list, [True, False]):
    print(tensor_dims, some)
    run_test(tensor_dims, some)
    print('-' * 20)