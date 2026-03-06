<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ACE-Step

[ACE-Step](https://github.com/ace-step/ACE-Step) is a music generation model that produces high-quality stereo audio at 48kHz from text prompts and optional lyrics. It uses a DiT architecture with flow matching, conditioned on text descriptions, lyrics, and timbre features.

The model comprises a text encoder (Qwen3-Embedding-0.6B), a 1D audio VAE (AutoencoderOobleck), and a transformer-based diffusion model with built-in lyric and timbre encoders. It supports musical metadata conditioning including BPM, time signature, and key.

## Tips

When constructing a prompt, keep in mind:

* Use descriptive genre and mood tags (e.g. "upbeat electronic dance music with a driving bassline").
* Musical metadata (`bpm`, `time_signature`, `key_scale`) can help steer the generation.
* Lyrics use section markers like `[verse]`, `[chorus]`, `[bridge]`. If no lyrics are provided, instrumental music is generated.

During inference:

* The turbo variant (`ACE-Step-v1-5-turbo`) produces good results in as few as 8 steps.
* The base variant uses more steps (e.g. 50) for higher quality.
* `audio_duration_in_s` controls the length of the generated audio (up to ~60s).

```py
import torch
import soundfile as sf
from diffusers import AceStepPipeline

pipe = AceStepPipeline.from_pretrained("ace-step/ACE-Step-v1-5-turbo", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

audio = pipe(
    prompt="upbeat electronic dance music",
    lyrics="[verse]\nDancing in the night\nFeeling so alive\n",
    audio_duration_in_s=10.0,
    num_inference_steps=8,
).audios

sf.write("output.wav", audio[0].T.float().cpu().numpy(), 48000)
```

## AceStepPipeline

[[autodoc]] AceStepPipeline
	- all
	- __call__
