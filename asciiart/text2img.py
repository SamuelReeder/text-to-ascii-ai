"""Text -> image with a distilled few-step diffusion model.

The prompt is steered toward ASCII-friendly pictures (one centered subject, bold shapes,
plain background). Several candidates are generated and the one whose *ASCII rendering*
best matches the prompt (by CLIP) is kept, so text prompts come out legible consistently.
"""
import torch
import torch.nn.functional as F

MODELS = {  # name: (repo, pinned revision)
    "sd-turbo": ("stabilityai/sd-turbo", "b261bac6fd2cf515557d5d0707481eafa0485ec2"),
    "sana-sprint": ("Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers", "aa76e7f4f4928f378716b6716a2130fba3caf5b1"),
    "flux2-klein": ("black-forest-labs/FLUX.2-klein-4B", "e7b7dc27f91deacad38e78976d1f2b499d76a294"),
}
STYLES = {
    "bold": ("{p}, centered, full view, simple bold illustration, clean thick outlines, high contrast, "
             "plain white background"),
    "flat": ("{p}, simple flat vector illustration, bold solid shapes, thick clean outlines, minimal detail, "
             "centered, full view, plain white background"),
}
STYLE = STYLES["flat"]


class TextToImage:
    def __init__(self, model: str = "sana-sprint", device: str = "cuda", keep_text_encoder: bool = True):
        """keep_text_encoder=False frees the (sana-sprint) text encoder after the first prompt is
        encoded, lowering peak VRAM from ~8.8 to ~4 GB when only one prompt will be used."""
        import diffusers
        repo, rev = MODELS[model]
        self.name, self.device, self.repo, self.rev = model, device, repo, rev
        self.keep_text_encoder, self._last = keep_text_encoder, (None, None)
        cuda = device.startswith("cuda")
        if model == "sd-turbo":
            self.dtype = torch.float16 if cuda else torch.float32
            self.pipe = diffusers.AutoPipelineForText2Image.from_pretrained(
                repo, revision=rev, torch_dtype=self.dtype, variant="fp16").to(device)
            self.kw = dict(num_inference_steps=2, guidance_scale=0.0, height=512, width=512)
        elif model == "sana-sprint":
            self.dtype = torch.bfloat16  # the Gemma-2 text encoder must stay in bf16
            self.pipe = diffusers.SanaSprintPipeline.from_pretrained(repo, revision=rev, torch_dtype=self.dtype).to(device)
            self.kw = dict(num_inference_steps=2, height=1024, width=1024)
        elif model == "flux2-klein":
            # text encoder (8 GB) and transformer (8 GB) don't fit together next to other apps:
            # encode prompts first, then swap in the transformer
            self.dtype = torch.bfloat16
            self.pipe, self._embeds = None, {}
            self.kw = dict(num_inference_steps=4, height=1024, width=1024)
        else:
            raise ValueError(model)
        if self.pipe is not None:
            self.pipe.set_progress_bar_config(disable=True)

    def encode(self, prompts: list[str]):
        """flux2-klein: precompute text embeddings for these prompts (styled) before generating."""
        if self.name != "flux2-klein":
            return
        import diffusers
        todo = [STYLE.format(p=p) for p in prompts if STYLE.format(p=p) not in self._embeds]
        if not todo:
            return
        self.pipe = None
        torch.cuda.empty_cache()
        te = diffusers.Flux2KleinPipeline.from_pretrained(self.repo, revision=self.rev, transformer=None, vae=None,
                                                          torch_dtype=self.dtype).to(self.device)
        with torch.no_grad():
            for p in todo:
                self._embeds[p] = te.encode_prompt(p, device=self.device)[0].cpu()
        del te
        torch.cuda.empty_cache()

    def _sana_embeds(self, p):
        """Encode once per prompt (not once per picture), with the pipeline's default instruction."""
        if self._last[0] != p:
            import inspect
            chi = inspect.signature(self.pipe.__call__).parameters["complex_human_instruction"].default
            self._last = (p, self.pipe.encode_prompt(p, device=self.device, complex_human_instruction=chi))
            if not self.keep_text_encoder:
                self.pipe.text_encoder = None
                torch.cuda.empty_cache()
        return self._last[1]

    @torch.no_grad()
    def __call__(self, prompt: str, n: int = 1, seed: int = 0, style: bool = True):
        p = STYLE.format(p=prompt) if style else prompt
        if self.name == "sd-turbo":
            g = torch.Generator(self.device).manual_seed(seed)
            return self.pipe([p] * n, generator=g, **self.kw).images
        # 1024 px models: one picture at a time (decoding a batch at 1024 px needs >10 GB)
        gens = [torch.Generator(self.device).manual_seed(seed * 1000 + k) for k in range(n)]
        if self.name == "sana-sprint":
            e, m = self._sana_embeds(p)
            return [self.pipe(prompt_embeds=e, prompt_attention_mask=m, generator=g, **self.kw).images[0]
                    for g in gens]
        if p not in self._embeds:
            self.encode([prompt] if style else [])
        if self.pipe is None:
            import diffusers
            self.pipe = diffusers.Flux2KleinPipeline.from_pretrained(
                self.repo, revision=self.rev, text_encoder=None, tokenizer=None, torch_dtype=self.dtype).to(self.device)
            self.pipe.set_progress_bar_config(disable=True)
        e = self._embeds[p].to(self.device)
        return [self.pipe(prompt_embeds=e, generator=g, **self.kw).images[0] for g in gens]


class PromptScorer:
    """CLIP similarity between a prompt and ASCII renderings (picks the most legible candidate)."""

    def __init__(self, spec=("ViT-B-16-SigLIP", "webli"), device="cuda"):
        import open_clip
        self.model, _, _ = open_clip.create_model_and_transforms(spec[0], pretrained=spec[1], device=device)
        self.model.eval()
        self.tok = open_clip.get_tokenizer(spec[0])
        cfg = self.model.visual.preprocess_cfg
        self.size = cfg["size"] if isinstance(cfg["size"], int) else cfg["size"][0]
        self.mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
        self.device = device

    @torch.no_grad()
    def __call__(self, prompt: str, renders: list[torch.Tensor]) -> torch.Tensor:
        t = F.normalize(self.model.encode_text(self.tok([f"ascii art of {prompt}"]).to(self.device)), dim=-1)
        xs = []
        for r in renders:
            x = r[None].to(self.device).float().expand(-1, 3, -1, -1)
            H, W = x.shape[-2:]
            S = max(H, W)
            x = F.pad(x, ((S - W) // 2, S - W - (S - W) // 2, (S - H) // 2, S - H - (S - H) // 2))
            xs.append(F.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True))
        x = (torch.cat(xs) - self.mean) / self.std
        e = F.normalize(self.model.encode_image(x), dim=-1)
        return (e @ t.T)[:, 0].cpu()
