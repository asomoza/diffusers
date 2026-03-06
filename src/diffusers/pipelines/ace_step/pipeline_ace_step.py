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

import math
from typing import Callable

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...models import AutoencoderOobleck
from ...models.transformers.transformer_ace_step import AceStepTransformer1DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline


logger = logging.get_logger(__name__)


def _apg_forward(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: list,
    momentum: float = -0.75,
    norm_threshold: float = 2.5,
):
    """Adaptive Projected Guidance (APG) — matches the original ACE-Step SFT guidance."""
    diff = pred_cond - pred_uncond

    # Momentum update
    if momentum_buffer:
        new_avg = momentum * momentum_buffer[0]
        momentum_buffer[0] = diff + new_avg
    else:
        momentum_buffer.append(diff.clone())
    diff = momentum_buffer[0]

    # Norm thresholding
    if norm_threshold > 0:
        diff_norm = diff.norm(p=2, dim=[1], keepdim=True)
        scale_factor = torch.minimum(torch.ones_like(diff_norm), norm_threshold / diff_norm)
        diff = diff * scale_factor

    # Project orthogonal to conditional prediction
    v0 = diff.double()
    v1 = torch.nn.functional.normalize(pred_cond.double(), dim=[1])
    v0_parallel = (v0 * v1).sum(dim=[1], keepdim=True) * v1
    diff_orthogonal = (v0 - v0_parallel).to(diff.dtype)

    return pred_cond + (guidance_scale - 1) * diff_orthogonal

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> import soundfile as sf
        >>> from diffusers import AceStepPipeline

        >>> pipe = AceStepPipeline.from_pretrained("ace-step/ACE-Step-v1-5-turbo", torch_dtype=torch.bfloat16)
        >>> pipe = pipe.to("cuda")

        >>> audio = pipe(
        ...     prompt="upbeat electronic dance music",
        ...     lyrics="[verse]\\nDancing in the night\\nFeeling so alive\\n",
        ...     audio_duration_in_s=10.0,
        ...     num_inference_steps=8,
        ... ).audios

        >>> sf.write("output.wav", audio[0].T.float().cpu().numpy(), 48000)
        ```
"""


class AceStepPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-music generation using ACE-Step.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderOobleck`]):
            Variational Auto-Encoder (VAE) model to encode and decode audio to and from latent representations.
        text_encoder ([`~transformers.PreTrainedModel`]):
            Frozen text-encoder (Qwen3-Embedding-0.6B).
        tokenizer ([`~transformers.PreTrainedTokenizerBase`]):
            Tokenizer for the text encoder.
        transformer ([`AceStepTransformer1DModel`]):
            ACE-Step DiT model to denoise the encoded audio latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler for flow matching denoising.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents"]

    # Template used during ACE-Step training for the text encoder input.
    _PROMPT_TEMPLATE = "# Instruction\n{instruction}\n\n# Caption\n{caption}\n\n# Metas\n{metas}<|endoftext|>\n"
    _LYRICS_TEMPLATE = "# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"
    _DEFAULT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"

    def __init__(
        self,
        vae: AutoencoderOobleck,
        text_encoder: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        transformer: AceStepTransformer1DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        # Cache silence_latent as a plain tensor so it remains accessible even when
        # sequential CPU offload moves transformer parameters to meta device.
        self._silence_latent = transformer.silence_latent.data.clone()

        self._vae_tiling_enabled = False
        self._vae_tile_chunk_size = 256
        self._vae_tile_overlap = 64

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    def enable_vae_tiling(self, chunk_size: int = 256, overlap: int = 64):
        r"""
        Enable tiled VAE decoding. The VAE decoder splits latents along the temporal axis into overlapping
        chunks, decodes each independently, trims the overlap, and concatenates. This drastically reduces
        peak VRAM during decode (the most memory-intensive step in the pipeline).

        Args:
            chunk_size (`int`, *optional*, defaults to 256):
                Size of each latent chunk in frames.
            overlap (`int`, *optional*, defaults to 64):
                Number of overlapping frames on each side of a chunk. These are decoded but discarded
                to avoid boundary artifacts from the convolutional decoder.
        """
        self._vae_tiling_enabled = True
        self._vae_tile_chunk_size = chunk_size
        self._vae_tile_overlap = overlap

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        decoding the full latent sequence in one pass.
        """
        self._vae_tiling_enabled = False

    def _tiled_vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents in overlapping temporal chunks to reduce VRAM usage.

        Uses overlap-discard chunking: each chunk is decoded with extra context on both sides,
        then only the core (non-overlapping) region is kept. This avoids boundary artifacts from
        the convolutional decoder while keeping peak VRAM proportional to chunk_size.

        Args:
            latents: Tensor of shape `[B, channels, latent_frames]`.

        Returns:
            Decoded audio tensor of shape `[B, audio_channels, samples]`.
        """
        chunk_size = self._vae_tile_chunk_size
        overlap = self._vae_tile_overlap
        latent_frames = latents.shape[-1]

        # Ensure overlap is valid for the chunk size
        while chunk_size - 2 * overlap <= 0 and overlap > 0:
            overlap = overlap // 2

        # If the sequence fits in one chunk, decode directly
        if latent_frames <= chunk_size:
            return self.vae.decode(latents).sample

        stride = chunk_size - 2 * overlap
        num_chunks = math.ceil(latent_frames / stride)

        decoded_chunks = []
        upsample_factor = None

        for i in range(num_chunks):
            core_start = i * stride
            core_end = min(core_start + stride, latent_frames)
            # Expand window by overlap on both sides for context
            win_start = max(0, core_start - overlap)
            win_end = min(latent_frames, core_end + overlap)

            latent_chunk = latents[:, :, win_start:win_end]
            audio_chunk = self.vae.decode(latent_chunk).sample

            # Compute the empirical upsample ratio from the first chunk
            if upsample_factor is None:
                upsample_factor = audio_chunk.shape[-1] / latent_chunk.shape[-1]

            # Trim the overlap regions in audio space
            added_start = core_start - win_start
            trim_start = int(round(added_start * upsample_factor))
            added_end = win_end - core_end
            trim_end = int(round(added_end * upsample_factor))

            audio_len = audio_chunk.shape[-1]
            end_idx = audio_len - trim_end if trim_end > 0 else audio_len
            decoded_chunks.append(audio_chunk[:, :, trim_start:end_idx])

        return torch.cat(decoded_chunks, dim=-1)

    def check_inputs(
        self,
        prompt,
        audio_duration_in_s,
        num_inference_steps,
        guidance_scale,
        callback_on_step_end_tensor_inputs=None,
    ):
        if prompt is None:
            raise ValueError("`prompt` must be provided.")
        if not isinstance(prompt, (str, list)):
            raise ValueError(
                f"`prompt` must be a string or list of strings, but got {type(prompt)}."
            )

        if audio_duration_in_s is not None and audio_duration_in_s <= 0:
            raise ValueError(
                f"`audio_duration_in_s` must be positive, but got {audio_duration_in_s}."
            )

        if num_inference_steps is not None and num_inference_steps <= 0:
            raise ValueError(
                f"`num_inference_steps` must be positive, but got {num_inference_steps}."
            )

        if guidance_scale is not None and guidance_scale < 1.0:
            raise ValueError(
                f"`guidance_scale` must be >= 1.0, but got {guidance_scale}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

    @staticmethod
    def _pad_and_batch_cfg(
        pos_tensor: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_tensor: torch.Tensor,
        neg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad positive and negative tensors to the same sequence length for CFG batching."""
        max_len = max(pos_tensor.shape[1], neg_tensor.shape[1])
        if pos_tensor.shape[1] < max_len:
            pad = max_len - pos_tensor.shape[1]
            pos_tensor = F.pad(pos_tensor, (0, 0, 0, pad))
            pos_mask = F.pad(pos_mask, (0, pad), value=0)
        if neg_tensor.shape[1] < max_len:
            pad = max_len - neg_tensor.shape[1]
            neg_tensor = F.pad(neg_tensor, (0, 0, 0, pad))
            neg_mask = F.pad(neg_mask, (0, pad), value=0)
        return pos_tensor, pos_mask, neg_tensor, neg_mask

    @staticmethod
    def _format_prompt(
        caption: str,
        audio_duration_in_s: float = 30.0,
        instruction: str | None = None,
        bpm: int | str | None = None,
        time_signature: int | str | None = None,
        key_scale: str | None = None,
    ) -> str:
        """Format a caption into the SFT prompt template expected by the text encoder.

        Args:
            caption: Text description of the desired music.
            audio_duration_in_s: Duration in seconds.
            instruction: Custom instruction (uses default if None).
            bpm: Beats per minute (e.g. 120). Defaults to "N/A".
            time_signature: Time signature numerator (e.g. 4 for 4/4). Defaults to "N/A".
            key_scale: Musical key (e.g. "A major", "E minor"). Defaults to "N/A".
        """
        if instruction is None:
            instruction = AceStepPipeline._DEFAULT_INSTRUCTION
        metas = (
            f"- bpm: {bpm if bpm is not None else 'N/A'}\n"
            f"- timesignature: {time_signature if time_signature is not None else 'N/A'}\n"
            f"- keyscale: {key_scale if key_scale is not None else 'N/A'}\n"
            f"- duration: {int(audio_duration_in_s)} seconds\n"
        )
        return AceStepPipeline._PROMPT_TEMPLATE.format(
            instruction=instruction,
            caption=caption,
            metas=metas,
        )

    @staticmethod
    def _format_lyrics(lyrics: str, language: str = "en") -> str:
        """Format lyrics into the template expected by the lyric encoder."""
        return AceStepPipeline._LYRICS_TEMPLATE.format(language=language, lyrics=lyrics)

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device,
        num_waveforms_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Encode text prompt using the text encoder."""
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            max_length=256,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        text_encoder_output = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        prompt_embeds = text_encoder_output.last_hidden_state

        # Duplicate for multiple waveforms per prompt
        if num_waveforms_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(
                num_waveforms_per_prompt, dim=0
            )
            attention_mask = attention_mask.repeat_interleave(
                num_waveforms_per_prompt, dim=0
            )

        negative_embeds = None
        negative_mask = None
        if do_classifier_free_guidance:
            if negative_prompt is not None:
                if isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt] * batch_size
                neg_inputs = self.tokenizer(
                    negative_prompt,
                    max_length=256,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                )
                neg_ids = neg_inputs.input_ids.to(device)
                neg_mask = neg_inputs.attention_mask.to(device)
                neg_output = self.text_encoder(
                    input_ids=neg_ids, attention_mask=neg_mask
                )
                negative_embeds = neg_output.last_hidden_state
                negative_mask = neg_mask
            else:
                negative_embeds = torch.zeros_like(prompt_embeds)
                negative_mask = torch.zeros_like(attention_mask)

            if num_waveforms_per_prompt > 1:
                negative_embeds = negative_embeds.repeat_interleave(
                    num_waveforms_per_prompt, dim=0
                )
                negative_mask = negative_mask.repeat_interleave(
                    num_waveforms_per_prompt, dim=0
                )

        return prompt_embeds, attention_mask, negative_embeds, negative_mask

    def encode_lyrics(
        self,
        lyrics: str | list[str] | None,
        device: torch.device,
        batch_size: int,
        num_waveforms_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Encode lyrics by looking up token embeddings (no text encoder forward pass)."""
        if lyrics is None:
            # Return empty embeddings
            embed_dim = self.text_encoder.config.hidden_size
            lyric_embeds = torch.zeros(batch_size, 1, embed_dim, device=device)
            lyric_mask = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            neg_lyric_embeds = lyric_embeds if do_classifier_free_guidance else None
            neg_lyric_mask = lyric_mask if do_classifier_free_guidance else None
            if num_waveforms_per_prompt > 1:
                lyric_embeds = lyric_embeds.repeat_interleave(
                    num_waveforms_per_prompt, dim=0
                )
                lyric_mask = lyric_mask.repeat_interleave(
                    num_waveforms_per_prompt, dim=0
                )
                if neg_lyric_embeds is not None:
                    neg_lyric_embeds = neg_lyric_embeds.repeat_interleave(
                        num_waveforms_per_prompt, dim=0
                    )
                    neg_lyric_mask = neg_lyric_mask.repeat_interleave(
                        num_waveforms_per_prompt, dim=0
                    )
            return lyric_embeds, lyric_mask, neg_lyric_embeds, neg_lyric_mask

        if isinstance(lyrics, str):
            lyrics = [lyrics] * batch_size

        lyric_inputs = self.tokenizer(
            lyrics,
            max_length=2048,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        lyric_ids = lyric_inputs.input_ids.to(device)
        lyric_mask = lyric_inputs.attention_mask.to(device)

        # Only embedding lookup, no encoder forward
        embedding_layer = self.text_encoder.get_input_embeddings()
        lyric_embeds = embedding_layer(lyric_ids)

        if num_waveforms_per_prompt > 1:
            lyric_embeds = lyric_embeds.repeat_interleave(
                num_waveforms_per_prompt, dim=0
            )
            lyric_mask = lyric_mask.repeat_interleave(num_waveforms_per_prompt, dim=0)

        neg_lyric_embeds = None
        neg_lyric_mask = None
        if do_classifier_free_guidance:
            neg_lyric_embeds = torch.zeros_like(lyric_embeds)
            neg_lyric_mask = torch.zeros_like(lyric_mask)

        return lyric_embeds, lyric_mask, neg_lyric_embeds, neg_lyric_mask

    def encode_audio(
        self,
        batch_size: int,
        device: torch.device,
        timbre_ref_length: int = 750,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode reference audio for timbre conditioning.

        For text-to-music generation, uses the pre-computed silence latent as the reference audio.
        The timbre encoder produces a CLS token embedding from this reference.

        Args:
            batch_size: Effective batch size (batch_size * num_waveforms_per_prompt).
            device: Device for tensors.
            timbre_ref_length: Number of latent frames for the timbre reference (~15s at 50Hz).

        Returns:
            Tuple of (timbre_hidden_states, timbre_mask).
        """
        timbre_hidden_states = self._silence_latent[:, :timbre_ref_length, :].expand(
            batch_size, -1, -1
        )
        timbre_hidden_states = timbre_hidden_states.to(device)
        timbre_mask = None  # No padding in the silence latent reference
        return timbre_hidden_states, timbre_mask

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] = None,
        lyrics: str | list[str] | None = None,
        audio_duration_in_s: float = 30.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        negative_prompt: str | list[str] | None = None,
        num_waveforms_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str | None = "pt",
        return_dict: bool = True,
        callback_on_step_end: (
            Callable | PipelineCallback | MultiPipelineCallbacks | None
        ) = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        shift: float = 3.0,
        turbo_sigmas: list[float] | None = None,
        lyric_language: str = "en",
        bpm: int | str | None = None,
        time_signature: int | str | None = None,
        key_scale: str | None = None,
    ):
        r"""
        The call function to the pipeline for music generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide music generation (e.g. genre/mood description).
            lyrics (`str` or `list[str]`, *optional*):
                The lyrics for the music. Use section markers like `[verse]`, `[chorus]`.
                If None, instrumental music is generated.
            audio_duration_in_s (`float`, *optional*, defaults to 30.0):
                Duration of the generated audio in seconds.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 1.0):
                Classifier-free guidance scale. The original ACE-Step model does not use CFG during
                inference (guidance_scale=1.0). Values > 1.0 enable classifier-free guidance with
                unconditional (zero) embeddings as the negative condition.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts for negative guidance.
            num_waveforms_per_prompt (`int`, *optional*, defaults to 1):
                The number of waveforms to generate per prompt.
            generator (`torch.Generator`, *optional*):
                A torch generator for reproducibility.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents.
            output_type (`str`, *optional*, defaults to `"pt"`):
                Output format. `"pt"` for PyTorch tensor, `"np"` for NumPy array.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return an `AudioPipelineOutput` or a tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising step during inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int,
                timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors
                as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`list`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the
                list will be passed as `callback_kwargs` argument. You will only be able to include variables
                listed in the `._callback_tensor_inputs` attribute of your pipeline class.
            shift (`float`, *optional*, defaults to 3.0):
                Shift parameter for flow matching timestep schedule.
            turbo_sigmas (`list[float]`, *optional*):
                Custom sigma schedule for turbo mode. These are the final (post-shift) sigma values to use
                directly. If provided, overrides num_inference_steps.
            lyric_language (`str`, *optional*, defaults to `"en"`):
                Language of the lyrics (e.g. "en", "zh", "ja").
            bpm (`int` or `str`, *optional*):
                Beats per minute for the generated music (e.g. 120).
            time_signature (`int` or `str`, *optional*):
                Time signature numerator (e.g. 4 for 4/4 time).
            key_scale (`str`, *optional*):
                Musical key (e.g. "A major", "E minor").

        Examples:

        Returns:
            [`~pipelines.AudioPipelineOutput`] or `tuple`.
        """
        # 0. Check inputs
        self.check_inputs(
            prompt,
            audio_duration_in_s,
            num_inference_steps,
            guidance_scale,
            callback_on_step_end_tensor_inputs,
        )

        # 1. Determine batch size and device
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        device = self._execution_device
        self._guidance_scale = guidance_scale

        # 2. Format and encode text prompt using the training template
        if isinstance(prompt, str):
            formatted_prompts = [
                self._format_prompt(
                    prompt,
                    audio_duration_in_s,
                    bpm=bpm,
                    time_signature=time_signature,
                    key_scale=key_scale,
                )
            ]
        else:
            formatted_prompts = [
                self._format_prompt(
                    p,
                    audio_duration_in_s,
                    bpm=bpm,
                    time_signature=time_signature,
                    key_scale=key_scale,
                )
                for p in prompt
            ]

        prompt_embeds, prompt_mask, negative_embeds, negative_mask = self.encode_prompt(
            formatted_prompts,
            device,
            num_waveforms_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
        )

        # 3. Format and encode lyrics using the training template
        formatted_lyrics = None
        if lyrics is not None:
            if isinstance(lyrics, str):
                formatted_lyrics = [self._format_lyrics(lyrics, lyric_language)]
            else:
                formatted_lyrics = [
                    self._format_lyrics(l, lyric_language) for l in lyrics
                ]

        lyric_embeds, lyric_mask, neg_lyric_embeds, neg_lyric_mask = self.encode_lyrics(
            formatted_lyrics,
            device,
            batch_size,
            num_waveforms_per_prompt,
            self.do_classifier_free_guidance,
        )

        # 4. Prepare timbre conditioning from silence latent (used for text2music)
        effective_batch = batch_size * num_waveforms_per_prompt
        timbre_hidden_states, timbre_mask = self.encode_audio(effective_batch, device)

        # 5. Prepare raw condition inputs (will be encoded inside transformer.forward)
        # CFG uses the transformer's learned null_condition_emb in the encoded hidden state
        # space, so we only encode the positive conditions here. The unconditional pass uses
        # null_condition_emb directly, matching how the model was trained.
        text_hidden_states = prompt_embeds
        text_mask = prompt_mask
        lyric_hidden_states = lyric_embeds
        lyric_attention_mask = lyric_mask

        # Cast condition tensors to transformer dtype (text encoder may output float32)
        dtype = self.transformer.dtype
        text_hidden_states = text_hidden_states.to(dtype=dtype)
        lyric_hidden_states = lyric_hidden_states.to(dtype=dtype)
        timbre_hidden_states = timbre_hidden_states.to(dtype=dtype)

        # 6. Compute latent dimensions
        hop_length = self.vae.hop_length
        sample_rate = self.vae.config.sampling_rate
        num_frames = int(audio_duration_in_s * sample_rate / hop_length)
        latent_channels = self.transformer.config.out_channels

        # 7. Prepare noise latents
        if latents is None:
            latent_shape = (effective_batch, num_frames, latent_channels)
            latents = randn_tensor(
                latent_shape,
                generator=generator,
                device=device,
                dtype=prompt_embeds.dtype,
            )
        else:
            latents = latents.to(device)

        # 8. Prepare context using silence latent (not zeros — the model was trained
        # with pre-computed silence VAE latents as the default source conditioning)
        src_latents = (
            self._silence_latent[:, :num_frames, :]
            .expand(effective_batch, -1, -1)
            .clone()
        )
        src_latents = src_latents.to(device=device, dtype=latents.dtype)
        chunk_mask = torch.ones_like(latents)

        # 9. Build timestep schedule via the scheduler
        # ACE-Step uses linspace(1, 0, steps+1) with shift applied, matching the original
        # implementation. We compute sigmas explicitly to avoid the scheduler's default
        # sigma_min/sigma_max which produces a slightly different schedule.
        if turbo_sigmas is not None:
            sigmas = turbo_sigmas
        else:
            sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1).tolist()[:-1]
            if shift != 1.0:
                sigmas = [shift * s / (1 + (shift - 1) * s) for s in sigmas]
        self.scheduler.set_shift(1.0)
        self.scheduler.set_timesteps(sigmas=sigmas, device=device)

        timesteps = self.scheduler.timesteps
        num_steps = len(timesteps)
        self._num_timesteps = num_steps

        # 10. Denoising loop
        # Conditions are encoded via the first forward() call (triggers CPU offload hooks),
        # then cached for all subsequent steps. For CFG, the unconditional pass uses the
        # learned null_condition_emb — matching how the model was trained.
        encoder_hidden_states = None
        encoder_mask = None
        null_encoder_hidden_states = None
        apg_momentum_buffer = []

        with self.progress_bar(total=num_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                sigma = self.scheduler.sigmas[i]

                model_input = torch.cat([src_latents, chunk_mask, latents], dim=-1)
                t_batch = sigma.expand(model_input.shape[0])
                timestep_r = t_batch

                if encoder_hidden_states is not None:
                    # Use cached encoded conditions
                    noise_pred = self.transformer(
                        model_input,
                        t_batch,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_mask,
                        timestep_r=timestep_r,
                        return_dict=False,
                    )[0]
                else:
                    # First step: encode conditions via forward() (needed for CPU offload)
                    noise_pred = self.transformer(
                        model_input,
                        t_batch,
                        text_hidden_states=text_hidden_states,
                        text_mask=text_mask,
                        lyric_embeds=lyric_hidden_states,
                        lyric_mask=lyric_attention_mask,
                        timbre_hidden_states=timbre_hidden_states,
                        timbre_mask=timbre_mask,
                        timestep_r=timestep_r,
                        return_dict=False,
                    )[0]
                    encoder_hidden_states = self.transformer._encoded_hidden_states
                    encoder_mask = self.transformer._encoded_attention_mask
                    if self.do_classifier_free_guidance:
                        # Project null condition through condition_embedder to match
                        # the original, which applies condition_embedder to all encoder
                        # hidden states (both cond and null_cond) inside the decoder.
                        null_encoder_hidden_states = self.transformer.condition_embedder(
                            self.transformer.null_condition_emb.expand_as(
                                encoder_hidden_states
                            )
                        )

                if self.do_classifier_free_guidance:
                    # Separate unconditional pass (avoids 2x VRAM from batch doubling)
                    noise_pred_uncond = self.transformer(
                        model_input,
                        t_batch,
                        encoder_hidden_states=null_encoder_hidden_states,
                        encoder_attention_mask=encoder_mask,
                        timestep_r=timestep_r,
                        return_dict=False,
                    )[0]
                    noise_pred = _apg_forward(
                        noise_pred,
                        noise_pred_uncond,
                        guidance_scale,
                        apg_momentum_buffer,
                    )

                # Scheduler step (handles Euler ODE step including final step with sigma_next=0)
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                progress_bar.update()

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

        # 11. VAE decode
        if output_type != "latent":
            # VAE expects [B, channels, time]
            latents_for_vae = latents.transpose(1, 2)
            if self._vae_tiling_enabled:
                audio = self._tiled_vae_decode(latents_for_vae)
            else:
                audio = self.vae.decode(latents_for_vae).sample
        else:
            self.maybe_free_model_hooks()
            if not return_dict:
                return (latents,)
            return AudioPipelineOutput(audios=latents)

        if output_type == "np":
            audio = audio.cpu().float().numpy()

        self.maybe_free_model_hooks()

        if not return_dict:
            return (audio,)

        return AudioPipelineOutput(audios=audio)
