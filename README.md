# AsciiNet

Turn **any image** — or a **text prompt** — into ASCII art that people can actually recognize, at
any size from a 16-column thumbnail to a full terminal.

![original vs ascii-image-converter vs the pipeline vs AsciiNet](docs/examples/comparison.png)
<sub>Left to right: original · `ascii-image-converter` (what the old dataset used) · the hand-built
pipeline (AsciiNet's teacher) · **AsciiNet at 80 columns** · **AsciiNet at 32 columns**. Held-out images.</sub>

```
$ python ascii.py -p "a steaming cup of coffee" -w 64
                                _
                               _|
                             _/)\,
                             `) |^_
                            ||/ ^,^,
                            \|\  ^\|\
                             \|\   \/
                              `,\  /
                               |`
                   _.><^^^`````` ````^^>><._
                 /``<_.><^^^^`````^^^^>><._`,\;
                 |><|<<.,____@@@@@____,,.><><^|_.<>><_
                 |||||``^^^^>>>>>><^^^^``|||||@@_><<_|\
                 |||||||||:::::::::::::::::|||@,`   ^@|/
                 |||||||:::::::::::::::::::::@@`     @|\
                 ^||||||::::::::::::::::::::\@/    _'@/
                  \|||||::::::::::::::::::::@|__,>^`_/
                   ,|||||::::::::::::::::::|@@@@_.<^
           _.>^^`   \||||||::::::::::::::|@_><^`  _
         /`          \_||||::::::::::::||_/        `^<
         _            |@|||||:::::||||@@/`            |
         `<>._      ``^<_@>@@,|_|,,@@_>^           ,</`
            `^<`^>><.,,__`^^^>>>><<^^   __,,.>>_^><`
                  `^><.,___```````````___,.><^`
                          ````^^^^````
```

## Quick start

Python 3.12+ and a monospace font are required (`fonts-dejavu-core` on Ubuntu).
The default installation is an isolated CPU environment for image conversion:

```bash
./setup.sh
source .venv/bin/activate
python ascii.py photo.jpg --engine net -w 80
python ascii.py photo.jpg --engine net --color -o art.txt --png art.png
```

For text prompts on an NVIDIA GPU (CUDA 13 builds, including RTX 50-series):

```bash
./setup.sh --cuda --profile text
source .venv/bin/activate
python ascii.py -p "a lighthouse on a cliff" -w 80
```

Profiles: `image` (default), `pipeline` (teacher), `text` (SANA + ranking),
`all` (training/evaluation/tests), and `demo` (Gradio). Use `--venv PATH` for a
separate environment, or install the matching `requirements/*.txt` into an
existing environment after installing PyTorch 2.13 (and torchvision 0.28 for
profiles other than `image`). No Conda installation is required.

The **28M-parameter AsciiNet** downloads 112 MB of released weights from a
[pinned Hugging Face revision](https://huggingface.co/SamuelReeder/asciinet/tree/55c4588acef906144ab5982049c6adf7c22a54a8).
A local `checkpoints/asciinet.pt` takes precedence in the CLI. `--engine net`
keeps image-only installations from falling back to the optional teacher.
Text prompts additionally download SANA-Sprint and its text encoder (~7 GB)
and SigLIP; allow about 9 GB of GPU memory.

```python
from asciiart.neural import NeuralConverter
from asciiart.pipeline import Options

net = NeuralConverter(device="cpu")
print(net.text("photo.jpg", Options(cols=80)))
```

## Results

On an RTX 5080, AsciiNet converts held-out images in **13 ms at 32 columns** and
**24 ms at 80 columns**, versus 108 ms for its teacher at 80 columns (4.5× faster).
These are local measurements, not hosted-service latency guarantees.
[Benchmark records](docs/benchmarks/gpu-speed.json) · [CPU feasibility](docs/benchmarks/cpu-feasibility.json)

## Interactive demo

The [Gradio app](app.py) accepts an image or a text prompt. Images run on CPU;
prompts use SANA-Sprint, then AsciiNet and SigLIP to choose between two candidates.
It is prepared for Hugging Face ZeroGPU; deployment instructions and the current
account-access limitation are in [docs/hosting.md](docs/hosting.md).

```bash
./setup.sh --cuda --profile demo
source .venv/bin/activate
python app.py
```

For a local image-only demo, set `ASCIINET_IMAGE_ONLY=1 ASCIINET_DEVICE=cpu`.

## Development and reproduction

```bash
python -m pip install -r requirements/dev.txt
python -m pytest                         # fast, offline checks
python -m pytest -m integration -k neural # released/local weights; CPU or GPU
```

See the [complete two-phase training recipe](docs/reproducing.md), including exact
recorded settings and the limits of reproducing an unseeded historical run.
Training data, checkpoints, and scratch runs remain local and ignored by Git.

- [Architecture, experiments, and evaluation](docs/research.md)
- [CLI options](docs/cli.md)
- [Validation results](docs/validation.md)
- [Models, licenses, and training-data restrictions](docs/models-and-licenses.md)
- [Local artifacts and archival guidance](docs/local-artifacts.md)

`asciiart/` contains the converter, model, teacher, and text pipeline. `scripts/`
contains data preparation, training, and evaluation. `legacy/` preserves the
original MNIST experiment; `asciiart/experimental/` contains abandoned approaches.
Code and AsciiNet weights use GPL-3.0; dependencies retain their own licenses.
