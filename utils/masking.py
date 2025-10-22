import torch


class TriangularCausalMask:
    """Upper-triangular causal mask for self-attention.

    Builds a boolean mask of shape [B, 1, L, L] where True means masked.
    Compatible with SelfAttention_Family.FullAttention usage.
    """

    def __init__(self, B: int, L: int, device=None):
        device = device or torch.device("cpu")
        # Upper triangle (excluding diagonal) is masked (True)
        base = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
        self._mask = base.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class ProbMask:
    """Mask helper for ProbAttention.

    If no explicit causal constraint is used, returns all-False mask with
    shape [B, H, L_Q, S] matching the given scores tensor.
    """

    def __init__(self, B: int, H: int, L_Q: int, index, scores: torch.Tensor, device=None):
        device = device or scores.device if isinstance(scores, torch.Tensor) else torch.device("cpu")
        S = scores.size(-1) if isinstance(scores, torch.Tensor) else L_Q
        self._mask = torch.zeros(B, H, L_Q, S, dtype=torch.bool, device=device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask
