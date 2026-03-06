#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""
Convert ACE-Step 1.5 checkpoints to diffusers format.

Usage:
    python scripts/convert_ace_step_to_diffusers.py \
        --checkpoint_path /path/to/acestep/checkpoint \
        --text_encoder_path Qwen/Qwen3-Embedding-0.6B \
        --vae_path stabilityai/stable-audio-open-1.0 \
        --output_path /path/to/output \
        --dtype fp32
"""

import argparse
import os
import re

import torch

from diffusers import AutoencoderOobleck, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_ace_step import AceStepTransformer1DModel


# Key mappings from original ACE-Step → diffusers


def convert_encoder_key(key: str) -> str | None:
    """Convert encoder (condition encoder) key to diffusers format."""
    # text_projector
    if key.startswith("encoder.text_projector."):
        return key.replace("encoder.text_projector.", "text_projector.")

    # lyric_encoder
    if key.startswith("encoder.lyric_encoder."):
        remainder = key[len("encoder.lyric_encoder.") :]
        return convert_lyric_encoder_key(remainder)

    # timbre_encoder
    if key.startswith("encoder.timbre_encoder."):
        remainder = key[len("encoder.timbre_encoder.") :]
        return convert_timbre_encoder_key(remainder)

    return None


def convert_lyric_encoder_key(key: str) -> str | None:
    """Convert lyric encoder keys."""
    # embed_tokens → proj_in
    if key.startswith("embed_tokens."):
        return "lyric_encoder.proj_in." + key[len("embed_tokens.") :]

    # norm
    if key.startswith("norm."):
        return "lyric_encoder.norm." + key[len("norm.") :]

    # layers.{i}.input_layernorm → layers.{i}.norm1
    m = re.match(r"layers\.(\d+)\.input_layernorm\.(.*)", key)
    if m:
        return f"lyric_encoder.layers.{m.group(1)}.norm1.{m.group(2)}"

    # layers.{i}.post_attention_layernorm → layers.{i}.norm2
    m = re.match(r"layers\.(\d+)\.post_attention_layernorm\.(.*)", key)
    if m:
        return f"lyric_encoder.layers.{m.group(1)}.norm2.{m.group(2)}"

    # layers.{i}.self_attn.{q,k,v,o}_proj → layers.{i}.attn.to_{q,k,v,out.0}
    m = re.match(r"layers\.(\d+)\.self_attn\.([qkvo])_proj\.(.*)", key)
    if m:
        layer_idx, proj, rest = m.group(1), m.group(2), m.group(3)
        if proj == "o":
            return f"lyric_encoder.layers.{layer_idx}.attn.to_out.0.{rest}"
        else:
            return f"lyric_encoder.layers.{layer_idx}.attn.to_{proj}.{rest}"

    # layers.{i}.self_attn.q_norm / k_norm → layers.{i}.attn.norm_q / norm_k
    m = re.match(r"layers\.(\d+)\.self_attn\.(q_norm|k_norm)\.(.*)", key)
    if m:
        layer_idx, norm_type, rest = m.group(1), m.group(2), m.group(3)
        diffusers_norm = "norm_q" if norm_type == "q_norm" else "norm_k"
        return f"lyric_encoder.layers.{layer_idx}.attn.{diffusers_norm}.{rest}"

    # layers.{i}.mlp.down_proj → layers.{i}.ff.net.2
    m = re.match(r"layers\.(\d+)\.mlp\.down_proj\.(.*)", key)
    if m:
        return f"lyric_encoder.layers.{m.group(1)}.ff.net.2.{m.group(2)}"

    # layers.{i}.mlp.gate_proj / up_proj → handled via fusion (skip here, fused separately)
    m = re.match(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj)\.(.*)", key)
    if m:
        return None  # Will be fused

    return None


def convert_timbre_encoder_key(key: str) -> str | None:
    """Convert timbre encoder keys."""
    # embed_tokens → proj_in
    if key.startswith("embed_tokens."):
        return "timbre_encoder.proj_in." + key[len("embed_tokens.") :]

    # special_token → cls_token
    if key == "special_token":
        return "timbre_encoder.cls_token"

    # norm
    if key.startswith("norm."):
        return "timbre_encoder.norm." + key[len("norm.") :]

    # Layer mappings (same pattern as lyric encoder but with timbre_encoder prefix)
    m = re.match(r"layers\.(\d+)\.input_layernorm\.(.*)", key)
    if m:
        return f"timbre_encoder.layers.{m.group(1)}.norm1.{m.group(2)}"

    m = re.match(r"layers\.(\d+)\.post_attention_layernorm\.(.*)", key)
    if m:
        return f"timbre_encoder.layers.{m.group(1)}.norm2.{m.group(2)}"

    m = re.match(r"layers\.(\d+)\.self_attn\.([qkvo])_proj\.(.*)", key)
    if m:
        layer_idx, proj, rest = m.group(1), m.group(2), m.group(3)
        if proj == "o":
            return f"timbre_encoder.layers.{layer_idx}.attn.to_out.0.{rest}"
        else:
            return f"timbre_encoder.layers.{layer_idx}.attn.to_{proj}.{rest}"

    m = re.match(r"layers\.(\d+)\.self_attn\.(q_norm|k_norm)\.(.*)", key)
    if m:
        layer_idx, norm_type, rest = m.group(1), m.group(2), m.group(3)
        diffusers_norm = "norm_q" if norm_type == "q_norm" else "norm_k"
        return f"timbre_encoder.layers.{layer_idx}.attn.{diffusers_norm}.{rest}"

    m = re.match(r"layers\.(\d+)\.mlp\.down_proj\.(.*)", key)
    if m:
        return f"timbre_encoder.layers.{m.group(1)}.ff.net.2.{m.group(2)}"

    m = re.match(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj)\.(.*)", key)
    if m:
        return None  # Will be fused

    return None


def convert_decoder_key(key: str) -> str | None:
    """Convert decoder (DiT) key to diffusers format."""
    # proj_in.1.weight/bias → patch_embed.weight/bias
    if key.startswith("decoder.proj_in.1."):
        return "patch_embed." + key[len("decoder.proj_in.1.") :]

    # proj_out.1.weight/bias → unpatch.weight/bias
    if key.startswith("decoder.proj_out.1."):
        return "unpatch." + key[len("decoder.proj_out.1.") :]

    # norm_out
    if key.startswith("decoder.norm_out."):
        return "norm_out." + key[len("decoder.norm_out.") :]

    # scale_shift_table (output)
    if key == "decoder.scale_shift_table":
        return "output_scale_shift_table"

    # condition_embedder
    if key.startswith("decoder.condition_embedder."):
        return "condition_embedder." + key[len("decoder.condition_embedder.") :]

    # time_embed / time_embed_r
    for te_prefix in ["time_embed", "time_embed_r"]:
        orig_prefix = f"decoder.{te_prefix}."
        if key.startswith(orig_prefix):
            remainder = key[len(orig_prefix) :]
            return convert_time_embed_key(remainder, te_prefix)

    # DiT layers
    m = re.match(r"decoder\.layers\.(\d+)\.(.*)", key)
    if m:
        layer_idx, remainder = m.group(1), m.group(2)
        return convert_dit_layer_key(layer_idx, remainder)

    return None


def convert_time_embed_key(key: str, prefix: str) -> str | None:
    """Convert time embedding keys.

    Original structure:
        time_embed.linear_1.{w,b}  → time_embed.linear_1.{w,b}  (TimestepEmbedding)
        time_embed.linear_2.{w,b}  → time_embed.linear_2.{w,b}
        time_embed.time_proj.{w,b} → time_proj.{w,b}  (separate Linear)
    """
    # time_proj is separate in diffusers
    if key.startswith("time_proj."):
        proj_name = "time_proj" if prefix == "time_embed" else "time_proj_r"
        return f"{proj_name}.{key[len('time_proj.'):]}"

    # linear_1, linear_2 → time_embed.1.linear_1, time_embed.1.linear_2 (in nn.Sequential[1] = TimestepEmbedding)
    if key.startswith("linear_1."):
        return f"{prefix}.1.linear_1.{key[len('linear_1.'):]}"
    if key.startswith("linear_2."):
        return f"{prefix}.1.linear_2.{key[len('linear_2.'):]}"

    return None


def convert_dit_layer_key(layer_idx: str, key: str) -> str | None:
    """Convert DiT layer keys."""
    # scale_shift_table
    if key == "scale_shift_table":
        return f"transformer_blocks.{layer_idx}.scale_shift_table"

    # self_attn_norm → norm1
    if key.startswith("self_attn_norm."):
        return f"transformer_blocks.{layer_idx}.norm1.{key[len('self_attn_norm.'):]}"

    # cross_attn_norm → norm2
    if key.startswith("cross_attn_norm."):
        return f"transformer_blocks.{layer_idx}.norm2.{key[len('cross_attn_norm.'):]}"

    # mlp_norm → norm3
    if key.startswith("mlp_norm."):
        return f"transformer_blocks.{layer_idx}.norm3.{key[len('mlp_norm.'):]}"

    # self_attn.{q,k,v,o}_proj → attn1.to_{q,k,v,out.0}
    m = re.match(r"self_attn\.([qkvo])_proj\.(.*)", key)
    if m:
        proj, rest = m.group(1), m.group(2)
        if proj == "o":
            return f"transformer_blocks.{layer_idx}.attn1.to_out.0.{rest}"
        else:
            return f"transformer_blocks.{layer_idx}.attn1.to_{proj}.{rest}"

    # self_attn.q_norm / k_norm → attn1.norm_q / norm_k
    m = re.match(r"self_attn\.(q_norm|k_norm)\.(.*)", key)
    if m:
        norm_type, rest = m.group(1), m.group(2)
        diffusers_norm = "norm_q" if norm_type == "q_norm" else "norm_k"
        return f"transformer_blocks.{layer_idx}.attn1.{diffusers_norm}.{rest}"

    # cross_attn.{q,k,v,o}_proj → attn2.to_{q,k,v,out.0}
    m = re.match(r"cross_attn\.([qkvo])_proj\.(.*)", key)
    if m:
        proj, rest = m.group(1), m.group(2)
        if proj == "o":
            return f"transformer_blocks.{layer_idx}.attn2.to_out.0.{rest}"
        else:
            return f"transformer_blocks.{layer_idx}.attn2.to_{proj}.{rest}"

    # cross_attn.q_norm / k_norm → attn2.norm_q / norm_k
    m = re.match(r"cross_attn\.(q_norm|k_norm)\.(.*)", key)
    if m:
        norm_type, rest = m.group(1), m.group(2)
        diffusers_norm = "norm_q" if norm_type == "q_norm" else "norm_k"
        return f"transformer_blocks.{layer_idx}.attn2.{diffusers_norm}.{rest}"

    # mlp.down_proj → ff.net.2
    if key.startswith("mlp.down_proj."):
        return f"transformer_blocks.{layer_idx}.ff.net.2.{key[len('mlp.down_proj.'):]}"

    # mlp.gate_proj / up_proj → fused (skip, handled separately)
    m = re.match(r"mlp\.(gate_proj|up_proj)\.(.*)", key)
    if m:
        return None

    return None


def fuse_swiglu_weights(state_dict: dict, prefix: str, target_prefix: str) -> dict:
    """Fuse gate_proj and up_proj weights into a single SwiGLU projection.

    SwiGLU in diffusers does: hidden, gate = proj(x).chunk(2)
    Then: hidden * silu(gate)

    So first half = up_proj (value), second half = gate_proj (gate through silu).
    """
    result = {}
    gate_key = f"{prefix}gate_proj.weight"
    up_key = f"{prefix}up_proj.weight"

    if gate_key in state_dict and up_key in state_dict:
        gate_w = state_dict[gate_key]
        up_w = state_dict[up_key]
        # First half = up_proj (value being gated), second half = gate_proj (through SiLU)
        fused_w = torch.cat([up_w, gate_w], dim=0)
        result[f"{target_prefix}weight"] = fused_w

    return result


def convert_checkpoint(state_dict: dict) -> dict:
    """Convert full ACE-Step state dict to diffusers format."""
    new_state_dict = {}
    unmapped_keys = []
    for key, value in state_dict.items():
        new_key = None

        # null_condition_emb
        if key == "null_condition_emb":
            new_key = "null_condition_emb"

        # Encoder keys
        elif key.startswith("encoder."):
            new_key = convert_encoder_key(key)

        # Decoder keys
        elif key.startswith("decoder."):
            new_key = convert_decoder_key(key)

        # Skip tokenizer/detokenizer/rotary_emb (not needed in diffusers model)
        elif any(
            key.startswith(p)
            for p in ["tokenizer.", "detokenizer.", "decoder.rotary_emb."]
        ):
            continue
        elif any(
            key.startswith(p)
            for p in [
                "encoder.lyric_encoder.rotary_emb.",
                "encoder.timbre_encoder.rotary_emb.",
            ]
        ):
            continue

        if new_key is not None:
            new_state_dict[new_key] = value
        elif key not in state_dict:
            pass  # Already consumed
        else:
            # Check if this is a gate/up proj that needs fusion
            is_fuse_key = False
            for pattern in [
                r"encoder\.lyric_encoder\.layers\.(\d+)\.mlp\.(gate_proj|up_proj)\.",
                r"encoder\.timbre_encoder\.layers\.(\d+)\.mlp\.(gate_proj|up_proj)\.",
                r"decoder\.layers\.(\d+)\.mlp\.(gate_proj|up_proj)\.",
            ]:
                if re.match(pattern, key):
                    is_fuse_key = True
                    break

            if not is_fuse_key:
                unmapped_keys.append(key)

    # Now handle SwiGLU fusions

    # Lyric encoder layers
    for i in range(100):  # Iterate over possible layer indices
        prefix = f"encoder.lyric_encoder.layers.{i}.mlp."
        gate_key = f"{prefix}gate_proj.weight"
        if gate_key not in state_dict:
            break
        target = f"lyric_encoder.layers.{i}.ff.net.0.proj."
        fused = fuse_swiglu_weights(state_dict, prefix, target)
        new_state_dict.update(fused)

    # Timbre encoder layers
    for i in range(100):
        prefix = f"encoder.timbre_encoder.layers.{i}.mlp."
        gate_key = f"{prefix}gate_proj.weight"
        if gate_key not in state_dict:
            break
        target = f"timbre_encoder.layers.{i}.ff.net.0.proj."
        fused = fuse_swiglu_weights(state_dict, prefix, target)
        new_state_dict.update(fused)

    # DiT decoder layers
    for i in range(100):
        prefix = f"decoder.layers.{i}.mlp."
        gate_key = f"{prefix}gate_proj.weight"
        if gate_key not in state_dict:
            break
        target = f"transformer_blocks.{i}.ff.net.0.proj."
        fused = fuse_swiglu_weights(state_dict, prefix, target)
        new_state_dict.update(fused)

    if unmapped_keys:
        print(f"WARNING: {len(unmapped_keys)} unmapped keys:")
        for k in unmapped_keys[:20]:
            print(f"  {k}")
        if len(unmapped_keys) > 20:
            print(f"  ... and {len(unmapped_keys) - 20} more")

    return new_state_dict


def main():
    parser = argparse.ArgumentParser(
        description="Convert ACE-Step checkpoint to diffusers format"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the ACE-Step checkpoint directory or safetensors file.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Path or HF hub id for the text encoder (Qwen3-Embedding-0.6B).",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path or HF hub id for the VAE (AutoencoderOobleck). If not provided, tries to load from checkpoint.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for the converted pipeline.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp16", "bf16", "fp32"],
        help="Data type for saving the model.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push to HuggingFace Hub.",
    )
    parser.add_argument(
        "--hub_id",
        type=str,
        default=None,
        help="Hub repository ID.",
    )
    args = parser.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    # 1. Load original checkpoint
    print("Loading original checkpoint...")
    ckpt_path = args.checkpoint_path

    # If checkpoint_path looks like a Hub ID (contains "/" but isn't a local path), download it
    if "/" in ckpt_path and not os.path.exists(ckpt_path):
        from huggingface_hub import snapshot_download

        print(f"Downloading from HuggingFace Hub: {ckpt_path}...")
        ckpt_path = snapshot_download(ckpt_path)

    if os.path.isdir(ckpt_path):
        # Try to find safetensors or bin file
        from safetensors.torch import load_file

        safetensor_files = [
            f for f in os.listdir(ckpt_path) if f.endswith(".safetensors")
        ]
        if safetensor_files:
            state_dict = {}
            for sf_file in safetensor_files:
                state_dict.update(load_file(os.path.join(ckpt_path, sf_file)))
        else:
            bin_files = [f for f in os.listdir(ckpt_path) if f.endswith(".bin")]
            state_dict = {}
            for bf in bin_files:
                state_dict.update(
                    torch.load(os.path.join(ckpt_path, bf), map_location="cpu")
                )
    else:
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

    # 2. Convert state dict
    print("Converting state dict...")
    new_state_dict = convert_checkpoint(state_dict)

    # 3. Create transformer model
    print("Creating transformer model...")
    transformer = AceStepTransformer1DModel(
        in_channels=192,
        out_channels=64,
        hidden_size=2048,
        num_layers=24,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=6144,
        patch_size=2,
        text_hidden_dim=1024,
        num_lyric_encoder_layers=8,
        num_timbre_encoder_layers=4,
        timbre_hidden_dim=64,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        sliding_window=128,
        sample_rate=48000,
    )

    # Load silence_latent (pre-computed VAE encoding of silence)
    silence_latent_path = None
    if os.path.isdir(ckpt_path):
        candidate = os.path.join(ckpt_path, "silence_latent.pt")
        if os.path.exists(candidate):
            silence_latent_path = candidate
    if silence_latent_path is None:
        # Check parent directory
        parent = (
            os.path.dirname(ckpt_path) if not os.path.isdir(ckpt_path) else ckpt_path
        )
        candidate = os.path.join(parent, "silence_latent.pt")
        if os.path.exists(candidate):
            silence_latent_path = candidate

    if silence_latent_path is not None:
        print(f"Loading silence_latent from {silence_latent_path}...")
        silence_latent = torch.load(silence_latent_path, map_location="cpu")
        # Original shape: [1, channels, time] → transpose to [1, time, channels]
        if (
            silence_latent.ndim == 3
            and silence_latent.shape[1] < silence_latent.shape[2]
        ):
            silence_latent = silence_latent.transpose(1, 2)
        new_state_dict["silence_latent"] = silence_latent
    else:
        print(
            "WARNING: silence_latent.pt not found. The model will use zeros as the default silence latent."
        )

    # Load converted weights
    transformer.load_state_dict(new_state_dict, strict=True)

    transformer = transformer.to(dtype)

    # 4. Load text encoder and tokenizer
    print("Loading text encoder and tokenizer...")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.text_encoder_path, trust_remote_code=True
    )
    text_encoder = AutoModel.from_pretrained(
        args.text_encoder_path, trust_remote_code=True, torch_dtype=dtype
    )

    # 5. Load or create VAE
    print("Loading VAE...")
    if args.vae_path:
        vae = AutoencoderOobleck.from_pretrained(
            args.vae_path,
            subfolder="vae" if "/" in args.vae_path else None,
            torch_dtype=dtype,
        )
    else:
        # Try to load from the checkpoint directory
        vae_path = os.path.join(
            os.path.dirname(ckpt_path) if not os.path.isdir(ckpt_path) else ckpt_path,
            "vae",
        )
        if os.path.exists(vae_path):
            vae = AutoencoderOobleck.from_pretrained(vae_path, torch_dtype=dtype)
        else:
            raise ValueError(
                "No VAE found. Please provide --vae_path pointing to a directory with an AutoencoderOobleck model."
            )

    # 6. Create scheduler
    print("Creating scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)

    # 7. Create and save pipeline
    print("Creating pipeline...")
    from diffusers.pipelines.ace_step.pipeline_ace_step import AceStepPipeline

    pipe = AceStepPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler,
    )

    print(f"Saving pipeline to {args.output_path}...")
    pipe.save_pretrained(args.output_path)

    if args.push_to_hub and args.hub_id:
        print(f"Pushing to hub: {args.hub_id}...")
        pipe.push_to_hub(args.hub_id)

    print("Done!")


if __name__ == "__main__":
    main()
