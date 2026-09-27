"""Gradio layout, kept separate from model startup for easy local checks."""
from pathlib import Path

import gradio as gr

from .engine import WIDTHS

CSS = """
#ascii-output textarea { font-family: ui-monospace, 'DejaVu Sans Mono', monospace;
  white-space: pre; overflow-x: auto; font-size: 12px; line-height: 1.15;
  background: #10151b; color: #edf4f8; }
.gradio-container { max-width: 1080px !important; }
"""


def build_demo(convert_image, generate_prompt=None):
    examples = Path(__file__).parent / "examples"
    with gr.Blocks(title="AsciiNet", analytics_enabled=False, delete_cache=(300, 300)) as demo:
        gr.Markdown("# AsciiNet\nTurn a picture or a few words into ASCII art.")
        gr.Markdown("A 28M-parameter model trained from scratch by [Samuel Reeder](https://samuelreeder.com). "
                    "[Source](https://github.com/SamuelReeder/asciinet) · "
                    "[Model](https://huggingface.co/SamuelReeder/asciinet)")
        with gr.Row():
            with gr.Column():
                width = gr.Dropdown(list(WIDTHS), value=80, label="Columns")
                with gr.Tab("From an image"):
                    upload = gr.File(label="Your image · up to 10 MB", type="filepath",
                                     file_types=[".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"])
                    convert = gr.Button("Convert image", variant="primary")
                    gr.Examples([[str(examples / (name + ".jpg"))] for name in ("butterfly", "helicopter", "octopus")],
                                inputs=[upload], cache_examples=False, label="Try an example")
                with gr.Tab("From words"):
                    if generate_prompt is not None:
                        prompt = gr.Textbox(label="Describe one subject", placeholder="a lighthouse on a cliff", max_length=500)
                        seed = gr.Number(value=0, minimum=0, maximum=2147483647, precision=0, label="Seed")
                        generate = gr.Button("Generate ASCII", variant="primary")
                        gr.Examples([["a butterfly"], ["a helicopter"], ["an octopus"]], inputs=[prompt], cache_examples=False)
                        gr.Markdown("SANA-Sprint creates two pictures; AsciiNet converts them and SigLIP picks the closer match. "
                                    "Shared GPU access may involve a queue or a daily usage limit.")
                    else:
                        gr.Markdown("Text generation is disabled in this local image-only session.")
            with gr.Column():
                picture = gr.Image(label="Source image", interactive=False, height=280, format="png")
                result = gr.Textbox(label="ASCII art", lines=24, max_lines=32, interactive=False,
                                    elem_id="ascii-output", buttons=["copy"])
        gr.Markdown("Inputs are processed on the host running this demo (Hugging Face for a Space). "
                    "The app does not save inputs as a dataset; temporary upload/output files are cleaned periodically. "
                    "GIFs use their first frame. This research demo works best with one clear subject.")
        convert.click(convert_image, [upload, width], [picture, result], api_name="image_to_ascii",
                      concurrency_limit=1, concurrency_id="inference")
        if generate_prompt is not None:
            generate.click(generate_prompt, [prompt, width, seed], [picture, result], api_name="text_to_ascii",
                           concurrency_limit=1, concurrency_id="inference")
    return demo.queue(max_size=8, default_concurrency_limit=1)
