"""Compare text->image models by how legible their prompt -> ASCII results are.

  gen:   generate --cands pictures for each prompt (Caltech-101 object names), time it, then convert
         every candidate to ASCII and keep the one SigLIP ranks best (exactly what ascii.py -p does)
  judge: ask the VLM judge (llama-server, see scripts/vlm_judge.py) which of 5 objects the ASCII art
         shows, and the same question about the picture itself

usage: python scripts/t2i_bench.py gen --model sana-sprint
       python scripts/t2i_bench.py judge --models sd-turbo sana-sprint flux2-klein
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OUT = Path("runs/t2i")


def prompts(n, seed=0):
    names = Path("data/raw/clip-benchmark__wds_vtab-caltech101/classnames.txt").read_text().strip().split("\n")
    names = [c for c in names if c not in ("background", "faces", "off-center face", "centered face")]
    rng = random.Random(seed)
    picked = rng.sample(names, n)
    return [{"i": i, "name": c, "prompt": f"a {c}",
             "options": rng.sample([o for o in names if o != c], 4)} for i, c in enumerate(picked)]


def gen(args):
    from asciiart.text2img import TextToImage
    ps = prompts(args.n)
    d = OUT / args.model
    d.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    t2i = TextToImage(args.model)
    t2i.encode([p["prompt"] for p in ps])  # flux2-klein: text encoder pass before loading the transformer
    t2i(ps[0]["prompt"])  # load everything / warm up
    load_s = time.time() - t0
    times = []
    for p in ps:
        torch.cuda.synchronize()
        t = time.time()
        pics = t2i(p["prompt"], n=args.cands, seed=0)
        torch.cuda.synchronize()
        times.append(time.time() - t)
        for k, im in enumerate(pics):
            im.resize((512, 512), Image.LANCZOS).save(d / f"{p['i']:03d}_{k}.jpg", quality=95)
    stats = {"model": args.model, "load_s": load_s, "gen_s_per_prompt": sum(times) / len(times),
             "cands": args.cands, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    print(stats, flush=True)
    del t2i
    torch.cuda.empty_cache()
    convert(args.model, ps, args.cands, stats)


def convert(model, ps, cands, stats, engine="pipeline", cols_list=(80, 40)):
    """engine "pipeline" writes NNN_ascii{cols}.png; "net" (the trained AsciiNet) NNN_net{cols}.png."""
    from asciiart.core import render_ids
    from asciiart.pipeline import Converter, Options
    from asciiart.text2img import PromptScorer
    from asciiart.viz import ids_to_pil
    d = OUT / model
    if engine == "net":
        from asciiart.neural import NeuralConverter
        conv = NeuralConverter()
    else:
        conv = Converter(matte=True, selector=False)
    tag = "net" if engine == "net" else "ascii"
    scorer = PromptScorer(device=conv.device)
    for cols in cols_list:
        for p in ps:
            pics = [Image.open(d / f"{p['i']:03d}_{k}.jpg") for k in range(cands)]
            res = [conv.ids(im, Options(cols=cols)) for im in pics]
            renders = [(render_ids(i[None], conv.atlas.cpu())[0] / conv.atlas.max().cpu()).clamp(0, 1) for i in res]
            best = int(scorer(p["prompt"], renders).argmax())
            ids_to_pil(res[best], conv.atlas.cpu(), scale=2).save(d / f"{p['i']:03d}_{tag}{cols}.png")
            p[f"best{cols}" if engine != "net" else f"best_net{cols}"] = best
    if engine != "net":
        json.dump({"stats": stats, "prompts": ps}, open(d / "meta.json", "w"))


def convert_existing(args):
    """ASCII-convert whatever a (stopped) gen run finished: prompts with all candidates present."""
    ps = [p for p in prompts(args.n)
          if all((OUT / args.model / f"{p['i']:03d}_{k}.jpg").exists() for k in range(args.cands))]
    stats = {"model": args.model, "cands": args.cands, "note": f"{len(ps)} prompts"}
    convert(args.model, ps, args.cands, stats)


def convert_net(args):
    """Also convert the benchmark pictures with the trained AsciiNet (after `gen`)."""
    meta = json.load(open(OUT / args.model / "meta.json"))
    convert(args.model, meta["prompts"], args.cands, meta["stats"], engine="net", cols_list=(80, 40, 32))


def judge(args):
    from scripts.vlm_judge import PROMPT, PROMPT_ORIG, ask
    for model in args.models:
        meta = json.load(open(OUT / model / "meta.json"))
        row = {"model": model, **meta["stats"]}
        for kind in args.kinds:
            ok = []
            for p in meta["prompts"]:
                rng = random.Random(p["i"])
                opts = [p["name"]] + p["options"]
                rng.shuffle(opts)
                q = "\n".join(f"{'ABCDE'[i]}. {o}" for i, o in enumerate(opts))
                if kind == "picture":
                    img, prompt = Image.open(OUT / model / f"{p['i']:03d}_{p['best80']}.jpg"), PROMPT_ORIG.format(opts=q)
                else:
                    img, prompt = Image.open(OUT / model / f"{p['i']:03d}_{kind}.png"), PROMPT.format(opts=q)
                a = ask(args.url, img, prompt)
                m = re.search(r"[A-E]", a.upper())
                ok.append(bool(m) and opts["ABCDE".index(m.group(0))] == p["name"])
            row[kind] = sum(ok) / len(ok)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        with open(OUT / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "convert", "convert-net", "judge"])
    ap.add_argument("--kinds", nargs="+", default=["ascii80", "ascii40", "picture"],
                    help="judge: ascii80 ascii40 (pipeline), net80 net40 net32 (AsciiNet), picture")
    ap.add_argument("--model", default="sana-sprint")
    ap.add_argument("--models", nargs="+", default=["sd-turbo", "sana-sprint", "flux2-klein"])
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--cands", type=int, default=4)
    ap.add_argument("--url", default="http://127.0.0.1:8089")
    args = ap.parse_args()
    {"gen": gen, "convert": convert_existing, "convert-net": convert_net, "judge": judge}[args.cmd](args)
