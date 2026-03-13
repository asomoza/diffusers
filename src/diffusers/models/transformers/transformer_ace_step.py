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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import AttentionMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..attention_processor import Attention
from ..cache_utils import CacheMixin
from ..embeddings import TimestepEmbedding, Timesteps, apply_rotary_emb, get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


logger = logging.get_logger(__name__)


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
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

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
                        sliding_window if (layer_types and layer_types[i] == "sliding_attention") else None
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
            float_mask = _convert_padding_mask_to_3d_attention_mask(attention_mask, seq_len, hidden_states.dtype)

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
                hidden_states = layer(hidden_states, rotary_emb=rotary_emb, attention_mask=layer_mask)

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
                        sliding_window if (layer_types and layer_types[i] == "sliding_attention") else None
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
            float_mask = _convert_padding_mask_to_3d_attention_mask(attention_mask, seq_len, hidden_states.dtype)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, rotary_emb, float_mask, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, rotary_emb=rotary_emb, attention_mask=float_mask)

        hidden_states = self.norm(hidden_states)
        # Extract CLS token output
        cls_output = hidden_states[:, 0:1, :]
        return cls_output


@maybe_allow_in_graph
class AceStepTransformerBlock(nn.Module):
    """Transformer block with AdaLN modulation, self-attention with RoPE, cross-attention, and SwiGLU FFN."""

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
        # Learned AdaLN modulation base: 6 params (shift, scale, gate) x 2 (self-attn + ffn)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, hidden_size) / hidden_size**0.5)

        # 1. Self-attention
        self.norm1 = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attn1 = Attention(
            query_dim=hidden_size,
            heads=num_attention_heads,
            dim_head=head_dim,
            kv_heads=num_key_value_heads,
            bias=False,
            out_bias=False,
            qk_norm="rms_norm",
            processor=AceStepAttnProcessor(),
        )

        # 2. Cross-attention
        self.norm2 = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attn2 = Attention(
            query_dim=hidden_size,
            cross_attention_dim=hidden_size,
            heads=num_attention_heads,
            dim_head=head_dim,
            kv_heads=num_key_value_heads,
            bias=False,
            out_bias=False,
            qk_norm="rms_norm",
            processor=AceStepAttnProcessor(),
        )

        # 3. Feed-forward
        self.norm3 = RMSNorm(hidden_size, eps=rms_norm_eps)
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
        timestep_proj: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Extract AdaLN modulation parameters
        shift_msa, scale_msa, gate_msa, shift_ff, scale_ff, gate_ff = (self.scale_shift_table + timestep_proj).chunk(
            6, dim=1
        )

        # 1. Self-attention with AdaLN
        norm_hidden_states = self.norm1(hidden_states) * (1 + scale_msa) + shift_msa
        attn_output = self.attn1(
            norm_hidden_states,
            rotary_emb=rotary_emb,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + attn_output * gate_msa

        # 2. Cross-attention (plain pre-norm, no timestep modulation)
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward with AdaLN
        norm_hidden_states = self.norm3(hidden_states) * (1 + scale_ff) + shift_ff
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_ff

        return hidden_states


class AceStepTransformer1DModel(ModelMixin, CacheMixin, AttentionMixin, ConfigMixin):
    """
    ACE-Step Diffusion Transformer for music generation.

    Uses a DiT architecture with flow matching for audio synthesis at 48kHz stereo. The model includes
    lyric and timbre encoders for conditioning, and uses patching for efficient processing of long audio sequences.

    Parameters:
        in_channels (`int`, defaults to 192): Input channels (latent + src + mask concatenated).
        out_channels (`int`, defaults to 64): Output channels (latent dim).
        hidden_size (`int`, defaults to 2048): Hidden dimension of the transformer.
        num_layers (`int`, defaults to 24): Number of DiT transformer blocks.
        num_attention_heads (`int`, defaults to 16): Number of attention heads.
        num_key_value_heads (`int`, defaults to 8): Number of KV heads for GQA.
        head_dim (`int`, defaults to 128): Dimension per attention head.
        intermediate_size (`int`, defaults to 6144): FFN intermediate dimension.
        patch_size (`int`, defaults to 2): Patch size for input/output convolutions.
        text_hidden_dim (`int`, defaults to 1024): Text encoder output dimension.
        num_lyric_encoder_layers (`int`, defaults to 8): Number of lyric encoder layers.
        num_timbre_encoder_layers (`int`, defaults to 4): Number of timbre encoder layers.
        timbre_hidden_dim (`int`, defaults to 64): Timbre input dimension.
        rms_norm_eps (`float`, defaults to 1e-6): Epsilon for RMSNorm.
        rope_theta (`float`, defaults to 1000000.0): Base for rotary embeddings.
        sliding_window (`int`, defaults to 128): Sliding window size for alternating layers.
        sample_rate (`int`, defaults to 48000): Audio sample rate.
    """

    _supports_gradient_checkpointing = True
    _supports_group_offloading = True
    _no_split_modules = ["AceStepTransformerBlock"]
    _skip_layerwise_casting_patterns = ["patch_embed", "unpatch", "norm", "time_embed", "lyric_encoder", "timbre_encoder", "condition_embedder"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 192,
        out_channels: int = 64,
        hidden_size: int = 2048,
        num_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 6144,
        patch_size: int = 2,
        text_hidden_dim: int = 1024,
        num_lyric_encoder_layers: int = 8,
        num_timbre_encoder_layers: int = 4,
        timbre_hidden_dim: int = 64,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        sliding_window: int = 128,
        sample_rate: int = 48000,
    ):
        super().__init__()

        # Build layer_types pattern (odd layers = sliding, even layers = full)
        layer_types = ["sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(num_layers)]

        # Patch embed / unpatch
        self.patch_embed = nn.Conv1d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.unpatch = nn.ConvTranspose1d(hidden_size, out_channels, kernel_size=patch_size, stride=patch_size)

        # Timestep embeddings (t and t-r)
        # Original uses scale=1000, sin-then-cos order, divide by half (not half-1)
        self.time_embed = nn.Sequential(
            Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000),
            TimestepEmbedding(256, hidden_size, act_fn="silu"),
        )
        self.time_proj = nn.Linear(hidden_size, hidden_size * 6)
        self.time_embed_r = nn.Sequential(
            Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000),
            TimestepEmbedding(256, hidden_size, act_fn="silu"),
        )
        self.time_proj_r = nn.Linear(hidden_size, hidden_size * 6)

        # Text projector
        self.text_projector = nn.Linear(text_hidden_dim, hidden_size, bias=False)

        # Lyric encoder
        self.lyric_encoder = AceStepLyricEncoder(
            text_hidden_dim=text_hidden_dim,
            hidden_size=hidden_size,
            num_layers=num_lyric_encoder_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            sliding_window=sliding_window,
            layer_types=layer_types[:num_lyric_encoder_layers],
        )

        # Timbre encoder
        self.timbre_encoder = AceStepTimbreEncoder(
            timbre_hidden_dim=timbre_hidden_dim,
            hidden_size=hidden_size,
            num_layers=num_timbre_encoder_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            sliding_window=sliding_window,
            layer_types=layer_types[:num_timbre_encoder_layers],
        )

        # Condition embedder (projects packed encoder states)
        self.condition_embedder = nn.Linear(hidden_size, hidden_size, bias=True)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                AceStepTransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                    sliding_window=(sliding_window if layer_types[i] == "sliding_attention" else None),
                )
                for i in range(num_layers)
            ]
        )

        # Output
        self.norm_out = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.output_scale_shift_table = nn.Parameter(torch.randn(1, 2, hidden_size) / hidden_size**0.5)

        # Null condition embedding for CFG
        self.null_condition_emb = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.gradient_checkpointing = False

        # Silence latent: pre-computed VAE encoding of silence, used as the default
        # source conditioning for text2music generation and as timbre encoder input.
        # Shape: [1, max_frames, out_channels] — stored transposed from the original [1, C, T].
        self.register_buffer("silence_latent", torch.zeros(1, 15000, out_channels), persistent=True)

    @staticmethod
    def _pack_sequences(
        hidden1: torch.Tensor,
        hidden2: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack two sequences by concatenating and sorting valid tokens first."""
        hidden_cat = torch.cat([hidden1, hidden2], dim=1)
        mask_cat = torch.cat([mask1, mask2], dim=1)
        batch_size, seq_len, dim = hidden_cat.shape

        sort_idx = mask_cat.argsort(dim=1, descending=True, stable=True)
        hidden_packed = torch.gather(hidden_cat, 1, sort_idx.unsqueeze(-1).expand(batch_size, seq_len, dim))

        lengths = mask_cat.sum(dim=1)
        new_mask = torch.arange(seq_len, dtype=torch.long, device=hidden_cat.device).unsqueeze(0) < lengths.unsqueeze(
            1
        )
        return hidden_packed, new_mask

    def encode_conditions(
        self,
        text_hidden_states: torch.Tensor,
        text_mask: torch.Tensor,
        lyric_embeds: torch.Tensor,
        lyric_mask: torch.Tensor,
        timbre_hidden_states: torch.Tensor | None = None,
        timbre_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode all condition inputs and pack them into a single sequence.

        Args:
            text_hidden_states: Text encoder outputs [B, seq, 1024].
            text_mask: Text attention mask [B, seq].
            lyric_embeds: Lyric token embeddings [B, seq, 1024].
            lyric_mask: Lyric attention mask [B, seq].
            timbre_hidden_states: Optional timbre features [B, seq, 64].
            timbre_mask: Optional timbre mask [B, seq].

        Returns:
            Tuple of (encoder_hidden_states, encoder_attention_mask).
        """
        # Project text: 1024 → 2048
        text_hidden_states = self.text_projector(text_hidden_states)

        # Encode lyrics: 1024 → 2048 via 8-layer encoder
        lyric_hidden_states = self.lyric_encoder(lyric_embeds, attention_mask=lyric_mask)

        # Pack lyrics + text
        if timbre_hidden_states is not None:
            # Encode timbre → [B, 1, 2048]
            timbre_output = self.timbre_encoder(timbre_hidden_states, attention_mask=timbre_mask)
            timbre_out_mask = torch.ones(
                timbre_output.shape[0],
                timbre_output.shape[1],
                dtype=lyric_mask.dtype,
                device=lyric_mask.device,
            )
            # Pack: lyrics + timbre, then + text
            encoder_hidden_states, encoder_mask = self._pack_sequences(
                lyric_hidden_states,
                timbre_output,
                lyric_mask,
                timbre_out_mask,
            )
        else:
            encoder_hidden_states = lyric_hidden_states
            encoder_mask = lyric_mask

        encoder_hidden_states, encoder_mask = self._pack_sequences(
            encoder_hidden_states,
            text_hidden_states,
            encoder_mask,
            text_mask,
        )

        # Project packed condition embeddings
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)

        return encoder_hidden_states, encoder_mask

    @staticmethod
    def _make_sliding_window_mask(
        seq_len: int,
        window_size: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create a bidirectional sliding window attention mask as [B, seq, seq] for AceStepAttnProcessor."""
        indices = torch.arange(seq_len, device=device)
        diff = indices.unsqueeze(1) - indices.unsqueeze(0)
        valid = torch.abs(diff) <= window_size
        min_val = torch.finfo(dtype).min
        mask = torch.where(
            valid,
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), min_val, dtype=dtype, device=device),
        )
        # Expand to [B, seq, seq] — the processor will repeat for heads
        return mask.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        timestep_r: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        text_hidden_states: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        lyric_embeds: torch.Tensor | None = None,
        lyric_mask: torch.Tensor | None = None,
        timbre_hidden_states: torch.Tensor | None = None,
        timbre_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple:
        """
        Forward pass of the ACE-Step DiT.

        Args:
            hidden_states: Pre-concatenated input [B, T, 192] (latent + src + mask).
            timestep: Diffusion timestep [B].
            encoder_hidden_states: Pre-encoded condition embeddings [B, seq, hidden_size].
                If None, conditions are encoded from the raw inputs below.
            encoder_attention_mask: Condition mask [B, seq].
            timestep_r: Second timestep for mean-flow [B]. Defaults to timestep if None.
            attention_mask: Optional latent attention mask [B, T].
            text_hidden_states: Raw text encoder outputs [B, seq, text_hidden_dim].
            text_mask: Text attention mask [B, seq].
            lyric_embeds: Lyric token embeddings [B, seq, text_hidden_dim].
            lyric_mask: Lyric attention mask [B, seq].
            timbre_hidden_states: Timbre features [B, seq, timbre_hidden_dim].
            timbre_mask: Timbre mask [B, seq].
            return_dict: Whether to return a dict or tuple.

        Returns:
            `Transformer2DModelOutput` or `tuple`. When raw condition inputs are provided and
            `return_dict=False`, returns `(sample, encoder_hidden_states, encoder_attention_mask)`.
        """
        # Encode conditions from raw inputs if pre-encoded states not provided
        encoded_from_raw = encoder_hidden_states is None
        if encoded_from_raw:
            encoder_hidden_states, encoder_attention_mask = self.encode_conditions(
                text_hidden_states=text_hidden_states,
                text_mask=text_mask,
                lyric_embeds=lyric_embeds,
                lyric_mask=lyric_mask,
                timbre_hidden_states=timbre_hidden_states,
                timbre_mask=timbre_mask,
            )

        if timestep_r is None:
            timestep_r = timestep

        batch_size = hidden_states.shape[0]
        original_seq_len = hidden_states.shape[1]

        # Pad if sequence length is not divisible by patch_size
        patch_size = self.config.patch_size
        pad_length = 0
        if hidden_states.shape[1] % patch_size != 0:
            pad_length = patch_size - (hidden_states.shape[1] % patch_size)
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length), mode="constant", value=0)

        # Patch embed: [B, T, C] → transpose → Conv1d → transpose → [B, T//patch, hidden]
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        seq_len = hidden_states.shape[1]

        # Timestep embeddings: sinusoidal (fp32) → cast → MLP
        t_freq = self.time_embed[0](timestep).to(hidden_states.dtype)
        temb_t = self.time_embed[1](t_freq)
        tr_freq = self.time_embed_r[0](timestep - timestep_r).to(hidden_states.dtype)
        temb_r = self.time_embed_r[1](tr_freq)
        temb = temb_t + temb_r

        timestep_proj_t = self.time_proj(F.silu(temb_t))
        timestep_proj_r = self.time_proj_r(F.silu(temb_r))
        timestep_proj = (timestep_proj_t + timestep_proj_r).unflatten(1, (6, -1))

        # Convert encoder_attention_mask [B, seq] to [B, 1, seq] float mask for cross-attention
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = _convert_padding_mask_to_3d_attention_mask(
                encoder_attention_mask,
                encoder_hidden_states.shape[1],
                hidden_states.dtype,
            )

        # Compute RoPE for sequence
        rotary_emb = get_1d_rotary_pos_embed(
            self.config.head_dim,
            seq_len,
            theta=self.config.rope_theta,
            use_real=True,
            repeat_interleave_real=False,
        )
        rotary_emb = (
            rotary_emb[0].to(hidden_states.device),
            rotary_emb[1].to(hidden_states.device),
        )

        # Process through transformer blocks
        for block in self.transformer_blocks:
            # Build self-attention mask for sliding window blocks
            self_attn_mask = None
            if block.sliding_window is not None:
                self_attn_mask = self._make_sliding_window_mask(
                    seq_len,
                    block.sliding_window,
                    batch_size,
                    hidden_states.device,
                    hidden_states.dtype,
                )

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    timestep_proj,
                    rotary_emb,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    self_attn_mask,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    timestep_proj=timestep_proj,
                    rotary_emb=rotary_emb,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    attention_mask=self_attn_mask,
                )

        # Output AdaLN
        shift, scale = (self.output_scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale) + shift

        # Unpatch: [B, T//patch, hidden] → transpose → ConvTranspose1d → transpose → [B, T, out_channels]
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.unpatch(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        # Crop back to original sequence length
        hidden_states = hidden_states[:, :original_seq_len, :]

        if not return_dict:
            if encoded_from_raw:
                return (hidden_states, encoder_hidden_states, encoder_attention_mask)
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
