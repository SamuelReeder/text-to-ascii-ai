"""Image embeddings from open_clip models (used to rank candidate renderings)."""
import torch
import torch.nn.functional as F


class ClipEmbedder:
    def __init__(self, specs=(("ViT-B-16-SigLIP", "webli"),), device="cuda"):
        import open_clip
        self.models = []
        for name, pre in specs:
            m, _, _ = open_clip.create_model_and_transforms(name, pretrained=pre, device=device)
            m = m.eval().requires_grad_(False)
            cfg = m.visual.preprocess_cfg
            size = cfg["size"] if isinstance(cfg["size"], int) else cfg["size"][0]
            mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
            std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
            self.models.append((m, size, mean, std))

    @staticmethod
    def _square(x, size, pad_value=0.0):
        H, W = x.shape[-2:]
        S = max(H, W)
        x = F.pad(x, ((S - W) // 2, S - W - (S - W) // 2, (S - H) // 2, S - H - (S - H) // 2), value=pad_value)
        return F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)

    def embed(self, x):
        """x (B,1|3,H,W) in [0,1] -> list of normalized embeddings, one per model."""
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        out = []
        for m, size, mean, std in self.models:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = m.encode_image((self._square(x, size) - mean) / std)
            out.append(F.normalize(e.float(), dim=-1))
        return out
