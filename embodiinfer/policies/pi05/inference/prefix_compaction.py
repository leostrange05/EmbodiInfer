"""Safe prefix padding compaction for the validated pi0.5 NPU profile."""

from __future__ import annotations

import torch


def compact_pi05_prefix(
    embeddings: torch.Tensor,
    padding_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    bucket: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove validated suffix padding or return the original tensors unchanged.

    The currently measured profile uses one sequence of 712 tokens and a 576
    bucket. The compact path applies only when all valid tokens form a
    contiguous prefix ending no later than the bucket boundary. Unexpected
    shapes and masks safely retain the original layout.
    """
    if (
        bucket != 576
        or embeddings.ndim != 3
        or embeddings.shape[:2] != (1, 712)
        or padding_mask.shape != (1, 712)
        or attention_mask.shape != (1, 712)
    ):
        return embeddings, padding_mask, attention_mask
    pad = padding_mask.detach().to(device="cpu", dtype=torch.bool)
    valid = int(pad.sum().item())
    if valid == 0 or valid > bucket or not bool(pad[0, :valid].all()) or bool(pad[0, valid:].any()):
        return embeddings, padding_mask, attention_mask
    return (
        embeddings[:, :bucket].contiguous(),
        padding_mask[:, :bucket].contiguous(),
        attention_mask[:, :bucket].contiguous(),
    )
