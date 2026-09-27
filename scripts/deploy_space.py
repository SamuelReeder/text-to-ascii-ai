"""Build an allowlisted Space bundle; --deploy explicitly publishes it on ZeroGPU."""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_bundle(destination):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Choose an empty output directory.")
    destination.mkdir(parents=True, exist_ok=True)
    # Only reviewed app code and generated examples; no data, runs, checkpoints,
    # user images, environment files, or credentials can enter this bundle.
    files = [ROOT / "app.py", ROOT / "LICENSE"]
    files += sorted((ROOT / "asciiart").glob("*.py"))
    files += sorted((ROOT / "demo").glob("*.py"))
    files += sorted((ROOT / "demo/examples").glob("*.jpg"))
    files += [ROOT / "demo/examples/provenance.json"]
    files += sorted((ROOT / "requirements").glob("*.txt"))
    for source in files:
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(ROOT / "demo/SPACE_README.md", destination / "README.md")
    (destination / "requirements.txt").write_text("torch==2.13.0\ntorchvision==0.28.0\n-r requirements/demo.txt\n")
    (destination / "packages.txt").write_text("fonts-dejavu-core\n")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "app.py", "asciiart", "demo", "requirements", "scripts/deploy_space.py"], cwd=ROOT, text=True))
    (destination / "DEPLOYMENT.json").write_text(json.dumps({
        "repository": "https://github.com/SamuelReeder/asciinet",
        "source_revision": revision, "uncommitted_app_changes": dirty,
        "hardware": "zero-a10g", "model_revision": "55c4588acef906144ab5982049c6adf7c22a54a8",
    }, indent=2) + "\n")
    return destination


def deploy(bundle, repo_id):
    from huggingface_hub import HfApi, SpaceHardware
    if json.loads((Path(bundle) / "DEPLOYMENT.json").read_text())["uncommitted_app_changes"]:
        raise ValueError("Commit app changes before deploying so the published source is reproducible.")
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=False,
                    space_hardware=SpaceHardware.ZERO_A10G, exist_ok=True)
    runtime = api.get_space_runtime(repo_id)
    if runtime.hardware != SpaceHardware.ZERO_A10G and runtime.requested_hardware != SpaceHardware.ZERO_A10G:
        api.request_space_hardware(repo_id, hardware=SpaceHardware.ZERO_A10G)
    return api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=bundle,
                             commit_message="Deploy AsciiNet image and text demo")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write a bundle to an empty directory")
    parser.add_argument("--deploy", action="store_true", help="publish the bundle to Hugging Face")
    parser.add_argument("--repo", default="SamuelReeder/asciinet")
    args = parser.parse_args()
    if not args.output and not args.deploy:
        parser.error("choose --output PATH or --deploy")
    with tempfile.TemporaryDirectory(prefix="asciinet-space-") as tmp:
        bundle = build_bundle(args.output or Path(tmp) / "space")
        if args.output:
            print(f"Bundle: {bundle.resolve()}")
        if args.deploy:
            try:
                print(deploy(bundle, args.repo))
            except Exception as exc:
                response = getattr(exc, "response", None)
                if response is not None and response.status_code in (401, 402, 403):
                    parser.exit(1, "Hugging Face denied access: " + str(response.json().get("error", "Check account permissions and ZeroGPU eligibility.")) + "\n")
                raise


if __name__ == "__main__":
    main()
