# Copyright 2025 The ACE-Step Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Encoder sub-modules and attention processor for ACE-Step."""

import torch
import torch.nn as nn

from ...models.attention import FeedForward
from ...models.attention_dispatch import dispatch_attention_fn
from ...models.attention_processor import Attention
from ...models.embeddings import apply_rotary_emb, get_1d_rotary_pos_embed
from ...models.normalization import RMSNorm


def _convert_padding_mask_to_3d_attention_mask(
    attention_mask: torch.Tensor, seq_len: int, dtype: torch.dtype
) -> torch.Tensor:
    """Convert a [B, seq] padding mask (1=valid, 0=pad) to a [B, 1, seq] additive attention mask.

    The processor reshapes this to [B, heads, 1, seq] for broadcast across query positions.
    """
    min_val = torch.finfo(dtype).min
    expanded = attention_mask.unsqueeze(1).to(dtype)
    return (1.0 - expanded) * min_val


class AceStepAttnProcessor:
    """Attention processor for ACE-Step that uses `dispatch_attention_fn` for backend flexibility.

    Supports RoPE (partial rotary embeddings), QK normalization, and GQA. Compatible with
    `model.set_attention_backend()` for runtime switching between Flash Attention, SageAttention, etc.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim

        # Reshape to [B, seq, heads, head_dim] for dispatch_attention_fn
        query = query.unflatten(2, (attn.heads, head_dim))
        key = key.unflatten(2, (kv_heads, head_dim))
        value = value.unflatten(2, (kv_heads, head_dim))

        # QK normalization (operates on [B, seq, heads, head_dim] via the last dim)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply partial RoPE (only to first rot_dim dimensions)
        if rotary_emb is not None:
            query_dtype = query.dtype
            key_dtype = key.dtype
            query = query.to(torch.float32)
            key = key.to(torch.float32)

            rot_dim = rotary_emb[0].shape[-1]
            query_to_rotate, query_unrotated = (
                query[..., :rot_dim],
                query[..., rot_dim:],
            )
            query_rotated = apply_rotary_emb(
                query_to_rotate,
                rotary_emb,
                use_real=True,
                use_real_unbind_dim=-2,
                sequence_dim=1,
            )
            query = torch.cat((query_rotated, query_unrotated), dim=-1)

            if not attn.is_cross_attention:
                key_to_rotate, key_unrotated = key[..., :rot_dim], key[..., rot_dim:]
                key_rotated = apply_rotary_emb(
                    key_to_rotate,
                    rotary_emb,
                    use_real=True,
                    use_real_unbind_dim=-2,
                    sequence_dim=1,
                )
                key = torch.cat((key_rotated, key_unrotated), dim=-1)

            query = query.to(query_dtype)
            key = key.to(key_dtype)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=kv_heads != attn.heads,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class AceStepLyricEncoderLayer(nn.Module):
    """Pre-norm transformer block for the lyric encoder."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        sliding_window: int | None = None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attn = Attention(
            query_dim=hidden_size,
            heads=num_attention_heads,
            dim_head=head_dim,
            kv_heads=num_key_value_heads,
            bias=False,
            out_bias=False,
            qk_norm="rms_norm",
            processor=AceStepAttnProcessor(),
        )
        self.norm2 = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.ff = FeedForward(
            hidden_size,
            activation_fn="swiglu",
            inner_dim=intermediate_size,
            bias=False,
        )
        self.sliding_window = sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(
            hidden_states,
            rotary_emb=rotary_emb,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class AceStepLyricEncoder(nn.Module):
    """Multi-layer encoder for lyrics using RoPE and bidirectional attention."""

    def __init__(
        self,
        text_hidden_dim: int,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        sliding_window: int | None = 128,
        layer_types: list[str] | None = None,
    ):
        super().__init__()
        self.proj_in = nn.Linear(text_hidden_dim, hidden_size)
        self.layers = nn.ModuleList(
            [
                AceStepLyricEncoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                    sliding_window=(
                        sliding_window
                        if (layer_types and layer_types[i] == "sliding_attention")
                        else None
                    ),
                )
                for i in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.proj_in(hidden_states)
        seq_len = hidden_states.shape[1]

        rotary_emb = get_1d_rotary_pos_embed(
            self.head_dim,
            seq_len,
            theta=self.rope_theta,
            use_real=True,
            repeat_interleave_real=False,
        )
        # Move to device
        rotary_emb = (
            rotary_emb[0].to(hidden_states.device),
            rotary_emb[1].to(hidden_states.device),
        )

        # Convert [B, seq] padding mask to [B, 1, seq] float attention mask for the processor
        batch_size = hidden_states.shape[0]
        float_mask = None
        if attention_mask is not None:
            float_mask = _convert_padding_mask_to_3d_attention_mask(
                attention_mask, seq_len, hidden_states.dtype
            )

        for layer in self.layers:
            # Build sliding window mask if needed
            layer_mask = float_mask
            if layer.sliding_window is not None:
                layer_mask = self._make_sliding_window_mask(
                    seq_len,
                    layer.sliding_window,
                    batch_size,
                    hidden_states.device,
                    hidden_states.dtype,
                    attention_mask,
                )

            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, rotary_emb, layer_mask, use_reentrant=False
                )
            else:
                hidden_states = layer(
                    hidden_states, rotary_emb=rotary_emb, attention_mask=layer_mask
                )

        hidden_states = self.norm(hidden_states)
        return hidden_states

    @staticmethod
    def _make_sliding_window_mask(
        seq_len: int,
        window_size: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Create a bidirectional sliding window mask as [B, seq, seq] additive attention mask."""
        indices = torch.arange(seq_len, device=device)
        diff = indices.unsqueeze(1) - indices.unsqueeze(0)
        valid = torch.abs(diff) <= window_size  # [seq, seq]

        if attention_mask is not None:
            # attention_mask: [B, seq] with 1=valid, 0=pad
            padding_mask = attention_mask.bool().unsqueeze(1)  # [B, 1, seq]
            valid = valid.unsqueeze(0) & padding_mask  # [B, seq, seq]
        else:
            valid = valid.unsqueeze(0).expand(batch_size, -1, -1)

        min_val = torch.finfo(dtype).min
        mask = torch.where(
            valid,
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), min_val, dtype=dtype, device=device),
        )
        return mask


class AceStepTimbreEncoder(nn.Module):
    """Multi-layer encoder with CLS token for timbre embedding extraction."""

    def __init__(
        self,
        timbre_hidden_dim: int,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        sliding_window: int | None = 128,
        layer_types: list[str] | None = None,
    ):
        super().__init__()
        self.proj_in = nn.Linear(timbre_hidden_dim, hidden_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.layers = nn.ModuleList(
            [
                AceStepLyricEncoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                    sliding_window=(
                        sliding_window
                        if (layer_types and layer_types[i] == "sliding_attention")
                        else None
                    ),
                )
                for i in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.proj_in(hidden_states)
        batch_size = hidden_states.shape[0]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        hidden_states = torch.cat([cls_tokens, hidden_states], dim=1)

        seq_len = hidden_states.shape[1]
        rotary_emb = get_1d_rotary_pos_embed(
            self.head_dim,
            seq_len,
            theta=self.rope_theta,
            use_real=True,
            repeat_interleave_real=False,
        )
        rotary_emb = (
            rotary_emb[0].to(hidden_states.device),
            rotary_emb[1].to(hidden_states.device),
        )

        # Convert [B, seq] padding mask to [B, 1, seq] float attention mask
        float_mask = None
        if attention_mask is not None:
            float_mask = _convert_padding_mask_to_3d_attention_mask(
                attention_mask, seq_len, hidden_states.dtype
            )

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, rotary_emb, float_mask, use_reentrant=False
                )
            else:
                hidden_states = layer(
                    hidden_states, rotary_emb=rotary_emb, attention_mask=float_mask
                )

        hidden_states = self.norm(hidden_states)
        # Extract CLS token output
        cls_output = hidden_states[:, 0:1, :]
        return cls_output
