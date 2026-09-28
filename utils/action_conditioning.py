"""Pack pixel-frame controls into the existing 32-channel action adapter.

The public inference path repeats each key into four adjacent channels. This
experimental extension assigns those channels consecutive pixel-frame samples:
channel = key_index * 4 + temporal_slot. Constant actions are exactly equivalent
to the original path; varying sub-frame semantics need checkpoint validation.
"""
from __future__ import annotations

import torch


def pack_frame_actions(
    actions, *, num_latent_frames: int, first_block: bool,
    height: int, width: int, device=None, dtype=torch.bfloat16,
) -> torch.Tensor:
    """Return [1, 32, latent_frames, height, width] without spatial copies.

    A fresh causal VAE decodes L latents to 1 + 4*(L-1) pixel frames.
    Replicate the reference frame's action into the first four slots. Later
    blocks decode L latents to 4*L frames. No temporal averaging is performed.
    """
    if num_latent_frames < 1 or height < 1 or width < 1:
        raise ValueError("Latent frame count and image dimensions must be positive.")
    values = torch.as_tensor(actions, dtype=torch.float32, device=device)
    expected = 4 * num_latent_frames - (3 if first_block else 0)
    if tuple(values.shape) != (expected, 8):
        raise ValueError(f"Expected actions [{expected}, 8], got {tuple(values.shape)}.")
    if not torch.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Action values must be finite and in [0, 1].")
    if first_block:
        values = torch.cat((values[:1].expand(3, -1), values), dim=0)
    packed = values.reshape(num_latent_frames, 4, 8).permute(2, 1, 0)
    packed = packed.reshape(1, 32, num_latent_frames, 1, 1).to(dtype=dtype)
    return packed.expand(-1, -1, -1, height, width)
