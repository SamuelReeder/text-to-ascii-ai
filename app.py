"""AsciiNet's local/ZeroGPU Gradio app. Import spaces before torch for CUDA emulation."""
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import spaces
import gradio as gr
import torch

from asciiart.neural import NeuralConverter, download_checkpoint
from demo.engine import image_to_ascii, prompt_to_ascii, validate_prompt, validate_width
from demo.ui import CSS, build_demo

torch.set_num_threads(2)
checkpoint = download_checkpoint()  # always use the published release, never local training weights
image_converter = NeuralConverter(checkpoint, device="cpu")


def convert_image(path, width):
    try:
        return image_to_ascii(image_converter, path, width)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


generate_prompt = None
if os.environ.get("ASCIINET_IMAGE_ONLY") != "1":
    from asciiart.text2img import PromptScorer, TextToImage

    device = os.environ.get("ASCIINET_DEVICE", "cuda")
    # ZeroGPU's emulation prepares these at module scope. Keep the encoder so
    # successive requests can use different prompts without reloading it.
    converter = NeuralConverter(checkpoint, device=device)
    generator = TextToImage("sana-sprint", device=device, keep_text_encoder=True)
    scorer = PromptScorer(device=device)

    @spaces.GPU(duration=60)
    def generate_on_gpu(prompt, width, seed):
        return prompt_to_ascii(converter, generator, scorer, prompt, width, seed)

    def generate_prompt(prompt, width, seed):
        try:
            prompt, seed = validate_prompt(prompt, seed)
            width = validate_width(width)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        return generate_on_gpu(prompt, width, seed)


demo = build_demo(convert_image, generate_prompt)
if __name__ == "__main__":
    demo.launch(css=CSS, max_file_size="10mb", show_error=False, run_history=False)
