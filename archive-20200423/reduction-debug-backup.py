import torch
dtype = torch.double
device = 'cuda'
a = torch.ones((), device='cuda')


def assert_allclose(actual, expected, rtol=1e-5, atol=1e-5, equal_nan=True, msg=''):
    if not isinstance(actual, torch.Tensor):
        actual = torch.tensor(actual)
    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected, dtype=actual.dtype)
    if expected.shape != actual.shape:
        expected = expected.expand_as(actual)

    close = torch.isclose(actual, expected, rtol, atol, equal_nan)
    print(close)
    if close.all():
        return

    # Find the worst offender
    error = (expected - actual).abs()
    # print(error)
    expected_error = atol + rtol * expected.abs()
    delta = error - expected_error
    delta[close] = 0  # mask out NaN/inf
    _, index = delta.reshape(-1).max(0)

    # TODO: consider adding torch.unravel_index
    def _unravel_index(index, shape):
        res = []
        for size in shape[::-1]:
            res.append(int(index % size))
            index = int(index // size)
        return tuple(res[::-1])

    index = _unravel_index(index.item(), actual.shape)

    # Count number of offenders
    count = (~close).long().sum()
    if msg == '' or msg is None:
        msg = ('Not within tolerance rtol={} atol={} at input{} ({} vs. {}) and {}'
                ' other locations ({:2.2f}%)')
        msg = msg.format(
            rtol, atol, list(index), actual[index].item(), expected[index].item(),
            count - 1, 100 * count / actual.numel())

    raise AssertionError(msg)

assert_allclose(a, a)
