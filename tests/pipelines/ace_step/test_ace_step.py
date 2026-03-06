# Copyright 2025 HuggingFace Inc.
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

import unittest

import torch
from transformers import AutoTokenizer, BertConfig, BertModel

from diffusers import AceStepPipeline, AutoencoderOobleck, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_ace_step import AceStepTransformer1DModel

from ...testing_utils import enable_full_determinism, torch_device
from ..pipeline_params import TEXT_TO_AUDIO_BATCH_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class AceStepPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = AceStepPipeline
    params = frozenset(
        [
            "prompt",
            "audio_duration_in_s",
            "guidance_scale",
            "negative_prompt",
            "lyrics",
        ]
    )
    batch_params = TEXT_TO_AUDIO_BATCH_PARAMS
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "num_waveforms_per_prompt",
            "generator",
            "latents",
            "output_type",
            "return_dict",
        ]
    )
    test_xformers_attention = False
    supports_dduf = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = AceStepTransformer1DModel(
            in_channels=18,
            out_channels=6,
            hidden_size=32,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            intermediate_size=64,
            patch_size=2,
            text_hidden_dim=32,
            num_lyric_encoder_layers=2,
            num_timbre_encoder_layers=2,
            timbre_hidden_dim=6,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            sliding_window=4,
            sample_rate=48000,
        )

        torch.manual_seed(0)
        vae = AutoencoderOobleck(
            encoder_hidden_size=6,
            downsampling_ratios=[1, 2],
            decoder_channels=6,
            decoder_input_channels=6,
            audio_channels=2,
            channel_multiples=[2, 4],
            sampling_rate=4,
        )

        scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)

        torch.manual_seed(0)
        text_encoder_config = BertConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=1000,
            max_position_embeddings=512,
        )
        text_encoder = BertModel(text_encoder_config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-BertModel")

        components = {
            "transformer": transformer,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "upbeat electronic dance music",
            "generator": generator,
            "num_inference_steps": 2,
            "audio_duration_in_s": 1.0,
        }
        return inputs

    def test_ace_step_basic(self):
        device = "cpu"
        components = self.get_dummy_components()
        pipe = AceStepPipeline(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = pipe(**inputs)
        audio = output.audios[0]

        assert audio.ndim == 2

    def test_ace_step_with_lyrics(self):
        device = "cpu"
        components = self.get_dummy_components()
        pipe = AceStepPipeline(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["lyrics"] = "[verse]\nHello world\n"
        output = pipe(**inputs)
        audio = output.audios[0]

        assert audio.ndim == 2

    def test_ace_step_negative_prompt(self):
        device = "cpu"
        components = self.get_dummy_components()
        pipe = AceStepPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["negative_prompt"] = "bad quality"
        inputs["guidance_scale"] = 3.0
        output = pipe(**inputs)
        audio = output.audios[0]

        assert audio.ndim == 2

    def test_ace_step_num_waveforms_per_prompt(self):
        device = "cpu"
        components = self.get_dummy_components()
        pipe = AceStepPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        prompt = "upbeat electronic dance music"

        # test num_waveforms_per_prompt=1 (default)
        audios = pipe(prompt, num_inference_steps=2, audio_duration_in_s=1.0).audios
        assert audios.shape[0] == 1

        # test num_waveforms_per_prompt=2
        num_waveforms_per_prompt = 2
        audios = pipe(
            prompt,
            num_inference_steps=2,
            audio_duration_in_s=1.0,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
        ).audios
        assert audios.shape[0] == num_waveforms_per_prompt

    def test_ace_step_latent_output(self):
        device = "cpu"
        components = self.get_dummy_components()
        pipe = AceStepPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["output_type"] = "latent"
        output = pipe(**inputs)
        latents = output.audios

        assert latents.ndim == 3

    def test_attention_slicing_forward_pass(self):
        self._test_attention_slicing_forward_pass(test_mean_pixel_difference=False)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=5e-4)

    def test_cpu_offload_forward_pass_twice(self):
        # Increase tolerance for composite model (lyric/timbre encoders + DiT)
        super().test_cpu_offload_forward_pass_twice(expected_max_diff=2e-3)

    @unittest.skip("Sequential offload produces NaN with this model's architecture.")
    def test_sequential_cpu_offload_forward_pass(self):
        pass

    @unittest.skip("Sequential offload produces NaN with this model's architecture.")
    def test_sequential_offload_forward_pass_twice(self):
        pass

    @unittest.skip("ACE-Step uses a custom attention processor not compatible with encode_prompt in isolation.")
    def test_encode_prompt_works_in_isolation(self):
        pass
