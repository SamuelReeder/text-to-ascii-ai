# Hosting the demo

[Back to AsciiNet](../README.md)

## Current deployment status

On 2026-09-27, Hugging Face authenticated `SamuelReeder` but rejected creation
of a ZeroGPU Space with HTTP 402: the account must have PRO, wait 30 days if new,
or obtain a community grant. No Space was created and no paid hardware was enabled.
The app and deployment bundle passed [local validation](validation.md); a hosted ZeroGPU
run remains unverified until Hugging Face enables access.

The [current ZeroGPU rules](https://huggingface.co/docs/hub/spaces-zerogpu) allow
up to two free Spaces for personal accounts in good standing with a verified
email and an account older than 30 days. The server's eligibility decision takes
precedence over this general documentation. A PRO subscription is a separate
account decision; the deployment command never purchases one or selects paid GPUs.

## Run locally

```bash
./setup.sh --cuda --profile demo
source .venv/bin/activate
python app.py
```

For CPU image conversion only:

```bash
ASCIINET_IMAGE_ONLY=1 ASCIINET_DEVICE=cpu python app.py
```

The UI supports uploads (PNG/JPEG/WebP/GIF/BMP) and prompts. It bounds files to
10 MB, decoded inputs to 16 megapixels and 1:4–4:1 aspect ratios, and output
widths to 32/64/80/120 columns. Prompts are limited to 500 characters and generate
two candidates. One shared inference slot and eight queue places bound work.
Image conversion uses CPU; prompt generation requests up to 60 seconds of ZeroGPU.
It keeps the SANA text encoder available for subsequent, different prompts.

The app imports `spaces` before torch and places prompt models on CUDA at module
scope, following ZeroGPU's CUDA-emulation rules. Compatibility with the hosted
runtime must still be checked. No `torch.compile` is used.

## Deploy when eligible

Authenticate with a Hugging Face identity that can create and write the Space.
Keep the token in the standard Hugging Face login/cache, never in this repository.

```bash
python scripts/deploy_space.py --output /tmp/asciinet-space-bundle
python scripts/deploy_space.py --deploy
```

`--output` creates a reviewable bundle without publishing. `--deploy` creates or
updates `SamuelReeder/asciinet` using **ZeroGPU only**. It uploads a strict allowlist:
app/runtime source, dependency profiles, license, and three generated examples.
It excludes local datasets, weights, scratch files, `cat.png`, and credentials.
`DEPLOYMENT.json` records the source revision. Commit app changes before deploying.
The Space README selects Python 3.12.12 and Gradio 6.28.0; requirements pin
PyTorch 2.13.0 and spaces 0.51.3. Linux font dependencies are included.

After publishing, verify that the Space reaches RUNNING on ZeroGPU, upload an
example, and run two different prompts sequentially. Verify quotas/errors and
mobile layout, then add a “Try AsciiNet” link or an on-demand embed to the portfolio.
Keep the saved sample gallery available while the Space sleeps or is quota-limited.
Do not add a live-demo link until the hosted app has passed those checks.

## Costs and visitor data

ZeroGPU has visitor queues and daily GPU quotas (currently two minutes for an
anonymous visitor). Image conversion remains CPU-only and needs no GPU allocation.
A paid CPU Space is unnecessary for this arrangement. See the
[Spaces overview](https://huggingface.co/docs/hub/spaces-overview).

Gradio manages temporary upload/output files and removes old files every five
minutes once they are at least five minutes old. The application does not write
inputs into a dataset or log prompt contents. This is not an assertion about
Hugging Face's own platform retention policies. Analytics is disabled in the app.
The interface explains that data is sent to the host before submission.
