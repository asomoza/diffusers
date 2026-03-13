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

import torch

from diffusers import AceStepTransformer1DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class AceStepTransformer1DTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AceStepTransformer1DModel

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (4, 8)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (4, 24)

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int] | tuple | str | bool]:
        return {
            "in_channels": 24,
            "out_channels": 8,
            "hidden_size": 32,
            "num_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 64,
            "patch_size": 2,
            "text_hidden_dim": 16,
            "num_lyric_encoder_layers": 2,
            "num_timbre_encoder_layers": 2,
            "timbre_hidden_dim": 8,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "sliding_window": 4,
            "sample_rate": 48000,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 1
        seq_len = 4
        in_channels = 24
        hidden_size = 32
        encoder_seq_len = 6

        return {
            "hidden_states": randn_tensor(
                (batch_size, seq_len, in_channels),
                generator=self.generator,
                device=torch_device,
            ),
            "timestep": torch.tensor([0.5], device=torch_device),
            "encoder_hidden_states": randn_tensor(
                (batch_size, encoder_seq_len, hidden_size),
                generator=self.generator,
                device=torch_device,
            ),
            "encoder_attention_mask": torch.ones(batch_size, encoder_seq_len, dtype=torch.long, device=torch_device),
        }


class TestAceStepTransformer1D(AceStepTransformer1DTesterConfig, ModelTesterMixin):
    """Core model tests for AceStepTransformer1DModel."""


class TestAceStepTransformer1DMemory(AceStepTransformer1DTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AceStepTransformer1DModel."""


class TestAceStepTransformer1DTraining(AceStepTransformer1DTesterConfig, TrainingTesterMixin):
    """Training tests for AceStepTransformer1DModel."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"AceStepTransformer1DModel", "AceStepLyricEncoder", "AceStepTimbreEncoder"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestAceStepTransformer1DAttention(AceStepTransformer1DTesterConfig, AttentionTesterMixin):
    """Attention processor tests for AceStepTransformer1DModel."""


class TestAceStepTransformer1DCompile(AceStepTransformer1DTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for AceStepTransformer1DModel."""
