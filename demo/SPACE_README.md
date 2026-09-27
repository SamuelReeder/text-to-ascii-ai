---
title: AsciiNet
emoji: 🦋
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: 3.12.12
app_file: app.py
pinned: false
license: gpl-3.0
short_description: A trained image-to-ASCII model, with SANA-Sprint text generation
models:
  - SamuelReeder/asciinet
  - Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers
  - timm/ViT-B-16-SigLIP
---

# AsciiNet

Turn pictures or text prompts into ASCII art. Images run on CPU; text generation
uses shared ZeroGPU access. Text mode generates two SANA-Sprint candidates,
converts both with AsciiNet, then ranks the ASCII renderings with SigLIP.

[Source and setup](https://github.com/SamuelReeder/asciinet) ·
[Model card](https://huggingface.co/SamuelReeder/asciinet) ·
[Samuel Reeder](https://samuelreeder.com)

Code and AsciiNet weights use GPL-3.0. Dependency licenses and training-data
restrictions are documented in the source repository and model card. This is a
research portfolio demo. Uploads are processed on Hugging Face; temporary files
are cleaned periodically and are not saved as a dataset by this application.

See `DEPLOYMENT.json` for the source commit used to build this Space.
