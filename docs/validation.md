# Cleanup and demo validation

Checked 2026-09-27 on Ubuntu/WSL, Python 3.12.

- Fresh CPU installation via `setup.sh --cpu --profile image`; no Conda, CUDA,
  transformers, diffusers, or torchvision required. CLI converted a curated example
  using the pinned released safetensors. CPU and demo environments pass `pip check`.
- 28 fast tests run with `HF_HUB_OFFLINE=1` and two CPU threads. They cover image
  modes, pinned model/config loading, absence of optional imports, CLI validation,
  upload limits before decoding, prompt validation/ranking, and deployment contents.
- The existing neural integration test passed on CPU across 15 image input types,
  three widths, two focus modes, and dark/light backgrounds. The teacher tests
  were not rerun; teacher inference was unchanged.
- The Gradio CPU endpoint converted the butterfly example. Chromium browser tests
  uploaded/converted it at 390 and 1440 pixels, with no page errors or horizontal
  page overflow. ASCII itself scrolls horizontally when needed.
- The full local Gradio endpoint generated two different prompts sequentially,
  confirming the text encoder remains usable across requests. Both returned a source
  image and nonempty ASCII. [Request measurements](benchmarks/demo-smoke.json)
  record 22.08 seconds cold and 1.80 seconds warm on this desktop GPU, not ZeroGPU.
- Deployment builds a reviewed bundle of app code, dependency profiles, license,
  and generated examples. It excludes local data/checkpoints/runs and the user's
  untracked `cat.png`. It refuses publication with uncommitted app changes and only
  requests ZeroGPU hardware.
- Hugging Face Space creation was attempted and rejected with HTTP 402 because
  account eligibility is not enabled. No Space was created, no subscription was
  purchased, and hosted ZeroGPU operation remains unverified. See [hosting](hosting.md).

Training was not rerun. Historical training settings and speed measurements were
preserved from the original checkpoints/logs, with reproduction limits documented.
