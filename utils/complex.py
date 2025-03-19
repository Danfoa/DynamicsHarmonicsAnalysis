import warnings

import torch


def interleave_with_conjugate(a: torch.Tensor):
    assert a.dtype == torch.cfloat or a.dtype == torch.cdouble
    new_shape = list(a.shape)
    if a.shape != 1:  # multi dimensional tensor
        d = a.shape[-1]
        new_shape[-1] = 2 * d
    else:
        d = 1
        new_shape = 2 * d

    a_conj_a = torch.concatenate([torch.unsqueeze(a, -1), torch.unsqueeze(torch.conj(a), -1)], dim=-1).view(new_shape)
    return a_conj_a


def view_as_complex(a: torch.Tensor):
    """Convert real valued tensors where the last dimension is expected to hold complex valued scalars flattened:
    i.e., the last dimension holds a = [Re(å_1), Im(å_1),...,Re(å_k), Im(å_k)]
    :param a: Real valued Tensor of dimension (..., 2k) being k the number of complex scalars in a
    :return: Å: Complex valued Tensor of dimension (..., k) holding Å=[å1,å2,...,åk]
    """
    if a.dtype == torch.cfloat or a.dtype == torch.cdouble:
        warnings.warn("Tensor is already complex valued, returning without change")

    if a.ndim >= 1:
        a_2d = a.view(a.shape[:-1] + (-1, 2))
    else:
        a_2d = a.view((-1, 2))

    a_complex = torch.view_as_complex(a_2d)
    return a_complex
