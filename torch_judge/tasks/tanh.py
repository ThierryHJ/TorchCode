"""Tanh activation, backward, and soft-capping task."""

TASK = {
    "title": "Tanh, Backward & Soft-Capping",
    "difficulty": "Easy",
    "function_name": "my_tanh",
    "hint": "tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)). The derivative is 1 - tanh(x)^2 — note it depends on the *output*, not the input. For soft-capping: cap * tanh(logits / cap).",
    "tests": [
        {
            "name": "Matches torch.tanh",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(4, 8)
out = {fn}(x)
ref = torch.tanh(x)
assert torch.allclose(out, ref, atol=1e-5), f'Does not match torch.tanh'
""",
        },
        {
            "name": "tanh(0) = 0 and bounded output",
            "code": """
import torch
out_zero = {fn}(torch.tensor([0.0]))
assert torch.allclose(out_zero, torch.tensor([0.0]), atol=1e-7), f'tanh(0) should be 0, got {out_zero.item()}'
x_large = torch.tensor([100., -100.])
out_large = {fn}(x_large)
assert (out_large.abs() <= 1.0 + 1e-5).all(), f'Output should be bounded in (-1, 1), got {out_large}'
""",
        },
        {
            "name": "Shape preservation",
            "code": """
import torch
x = torch.randn(2, 3, 4)
assert {fn}(x).shape == x.shape, 'Shape mismatch'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
x = torch.randn(4, 8, requires_grad=True)
{fn}(x).sum().backward()
assert x.grad is not None and x.grad.shape == x.shape, 'Gradient issue'
""",
        },
        {
            "name": "Manual backward (tanh_backward)",
            "code": """
import torch
x = torch.randn(4, 8, requires_grad=True)
out = {fn}(x)
out.sum().backward()
autograd_grad = x.grad.clone()

tanh_out = {fn}(x.detach())
grad_output = torch.ones_like(tanh_out)
manual_grad = tanh_backward(grad_output, tanh_out)
assert torch.allclose(manual_grad, autograd_grad, atol=1e-5), f'tanh_backward does not match autograd'
""",
        },
        {
            "name": "Soft-capping bounds logits",
            "code": """
import torch
logits = torch.tensor([-50., -10., 0., 10., 50.])
cap = 30.0
capped = soft_cap_logits(logits, cap)
assert (capped.abs() < cap).all(), f'Soft-capped output should be within (-{cap}, {cap}), got {capped}'
assert torch.allclose(capped[2], torch.tensor(0.0), atol=1e-7), f'soft_cap(0) should be 0, got {capped[2]}'
ref = cap * torch.tanh(logits / cap)
assert torch.allclose(capped, ref, atol=1e-5), f'Does not match cap * tanh(logits / cap)'
""",
        },
    ],
}
