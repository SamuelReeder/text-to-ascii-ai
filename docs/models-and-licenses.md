## Models and licenses

| model | used for | license |
|---|---|---|
| AsciiNet (this repo; weights on [Hugging Face](https://huggingface.co/SamuelReeder/asciinet)) | image → ASCII | this repo's license (GPL-3.0) |
| [SANA-Sprint 0.6B](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers) (pinned revision) + its [Gemma-2-2B](https://huggingface.co/google/gemma-2-2b-it) text encoder | text → picture (prompts only) | Apache-2.0 / Gemma Terms of Use |
| [SigLIP ViT-B/16](https://huggingface.co/timm/ViT-B-16-SigLIP) (open_clip) | prompt reranking, `--focus select` | Apache-2.0 |
| [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) (pinned revision; loads its own code via `trust_remote_code`) | the pipeline's subject matte; training labels | MIT |
| [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) | `--t2i sd-turbo` | Stability AI Non-Commercial Research Community License |
| Qwen3-VL-8B, CLIP ViT-L/14 (DataComp) | evaluation only | Apache-2.0 / MIT |

The training images come from the public datasets listed above, each under its own license;
they are used for training and evaluation only and are not redistributed.

[Back to AsciiNet](../README.md)
