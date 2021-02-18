
import torch
import torch.nn.functional as F
# from torch.autograd.gradcheck import get_analytical_jacobian
from torch._six import container_abcs, istuple
torch.backends.cudnn.enabled = False
# torch.backends.cudnn.deterministic = True

def iter_tensors(x, only_requiring_grad=False):
    if isinstance(x, torch.Tensor):
        if x.requires_grad or not only_requiring_grad:
            yield x
    elif isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
        for elem in x:
            for result in iter_tensors(elem, only_requiring_grad):
                yield result

def make_jacobian(input, num_out):
    if isinstance(input, torch.Tensor):
        if not input.is_floating_point():
            return None
        if not input.requires_grad:
            return None
        return torch.zeros(input.nelement(), num_out, dtype=input.dtype)
    elif isinstance(input, container_abcs.Iterable) and not isinstance(input, str):
        jacobians = list(filter(
            lambda x: x is not None, (make_jacobian(elem, num_out) for elem in input)))
        if not jacobians:
            return None
        return type(input)(jacobians)
    else:
        return None

def get_analytical_jacobian(input, output, nondet_tol=0.0):
    # it is easier to call to_dense() on the sparse output than
    # to modify analytical jacobian
    if output.is_sparse:
        raise ValueError('Sparse output is not supported at gradcheck yet. '
                         'Please call to_dense() on the output of fn for gradcheck.')
    if output.layout == torch._mkldnn:
        raise ValueError('MKLDNN output is not supported at gradcheck yet. '
                         'Please call to_dense() on the output of fn for gradcheck.')
    diff_input_list = list(iter_tensors(input, True))
    jacobian = make_jacobian(input, output.numel())
    jacobian_reentrant = make_jacobian(input, output.numel())
    grad_output = torch.zeros_like(output, memory_format=torch.legacy_contiguous_format)
    flat_grad_output = grad_output.view(-1)
    reentrant = True
    correct_grad_sizes = True

    for i in range(flat_grad_output.numel()):
        flat_grad_output.zero_()
        flat_grad_output[i] = 1
        for jacobian_c in (jacobian, jacobian_reentrant):
            grads_input = torch.autograd.grad(output, diff_input_list, grad_output,
                                              retain_graph=True, allow_unused=True)
            for jacobian_x, d_x, x in zip(jacobian_c, grads_input, diff_input_list):
                if d_x is not None and d_x.size() != x.size():
                    correct_grad_sizes = False
                elif jacobian_x.numel() != 0:
                    if d_x is None:
                        jacobian_x[:, i].zero_()
                    else:
                        d_x_dense = d_x.to_dense() if not d_x.layout == torch.strided else d_x
                        assert jacobian_x[:, i].numel() == d_x_dense.numel()
                        jacobian_x[:, i] = d_x_dense.contiguous().view(-1)

    for jacobian_x, jacobian_reentrant_x in zip(jacobian, jacobian_reentrant):
        if jacobian_x.numel() != 0 and (jacobian_x - jacobian_reentrant_x).abs().max() > nondet_tol:
            print((jacobian_x - jacobian_reentrant_x).abs().max())
            reentrant = False

    return jacobian, reentrant, correct_grad_sizes

def test_ctc_loss(device):
    batch_size = 64
    num_labels = 101
    input_length = 150

    with open('save.bin', 'rb') as f:
        targets, x, tile_factors, input_lengths, target_lengths = torch.load(f)

    def ctc_after_softmax(x):
        x_full = ((x[:, None] * tile_factors[None, :]).view(-1)[:input_length * batch_size * num_labels]
                    .view(input_length, batch_size, num_labels))
        log_probs = torch.log_softmax(x_full, 2)
        # print(log_probs.shape)
        return torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)

    input_ = (x.double(),)
    output_ = ctc_after_softmax(input_[0])

    _, reentrant, _ = get_analytical_jacobian(input_, output_, 1e-17)
    print(reentrant)

if __name__=='__main__':
    test_ctc_loss('cuda')
