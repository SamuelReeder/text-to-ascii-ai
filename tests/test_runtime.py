"""Offline contracts for minimal installations, releases, and public-demo inputs."""
import json
import subprocess
import sys

import pytest
import torch
from PIL import Image

from asciiart.core import grid_rows
from asciiart import neural
from demo.engine import (prompt_to_ascii, read_upload, validate_prompt,
                         validate_width, MAX_UPLOAD_BYTES)
from scripts.deploy_space import build_bundle


def test_release_download_is_one_pinned_snapshot(monkeypatch, tmp_path):
    import huggingface_hub
    requests = []
    def snapshot(repo, **kwargs):
        requests.append((repo, kwargs))
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot)
    assert neural.download_checkpoint() == tmp_path / 'model.safetensors'
    assert requests == [(neural.HF_REPO, {
        'revision': neural.HF_REVISION,
        'allow_patterns': ['config.json', 'model.safetensors'],
    })]
    assert len(neural.HF_REVISION) == 40


def test_local_checkpoint_does_not_download(monkeypatch, tmp_path):
    checkpoint = tmp_path / 'local.pt'
    checkpoint.touch()
    monkeypatch.setattr(neural, 'DEFAULT_CKPT', checkpoint)
    def unexpected_download():
        pytest.fail('Local weights must not require the network')
    monkeypatch.setattr(neural, 'download_checkpoint', unexpected_download)
    assert neural.find_checkpoint() == checkpoint


def test_safetensors_config_and_weights_load_together(tmp_path):
    from safetensors.torch import save_file
    path = tmp_path / 'model.safetensors'
    save_file({'weight': torch.tensor([1.0])}, path)
    (tmp_path / 'config.json').write_text(json.dumps({'chars': ' @', 'step': 20000}))
    state, chars, step = neural.load_checkpoint(path)
    assert chars == ' @' and step == 20000
    assert state['weight'].tolist() == [1.0]


def test_cli_image_runtime_does_not_import_optional_models():
    # A deliberate import blocker also catches regressions in a full dev environment.
    code = '''
import importlib.abc, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'diffusers', 'transformers', 'open_clip', 'timm', 'kornia'}:
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import ascii
ascii.quiet_libraries()
from asciiart.neural import NeuralConverter
'''
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('args', [['--width', '0'], ['--width', '-4'], ['--candidates', '0']])
def test_cli_rejects_invalid_sizes_before_loading_models(args):
    result = subprocess.run([sys.executable, 'ascii.py', 'unused.png', *args], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'must be between' in result.stderr


@pytest.mark.parametrize('width', [0, 16, 121, 512, '80', None])
def test_demo_rejects_unbounded_widths(width):
    with pytest.raises(ValueError):
        validate_width(width)


@pytest.mark.parametrize('prompt,seed', [('', 0), ('a' * 501, 0), ('cat', -1), ('cat', 1.5), ('cat', float('nan')), ('cat', 2**31)])
def test_demo_rejects_invalid_prompt_requests(prompt, seed):
    with pytest.raises(ValueError):
        validate_prompt(prompt, seed)


def test_upload_decodes_first_frame_and_bounds_grid(tmp_path):
    path = tmp_path / 'image.gif'
    Image.new('RGB', (100, 200), 'red').save(path, save_all=True, append_images=[Image.new('RGB', (100, 200), 'blue')])
    image = read_upload(path)
    assert image.mode == 'RGB' and image.getpixel((0, 0)) == (255, 0, 0)
    assert grid_rows(*image.size, 120) <= 240


@pytest.mark.parametrize('size', [(5000, 4000), (10, 100)])
def test_upload_rejects_large_or_extreme_images_before_decoding(tmp_path, monkeypatch, size):
    path = tmp_path / 'large.png'
    Image.new('1', size).save(path)
    def decode_would_be_too_late(*args, **kwargs):
        pytest.fail('Pixel/aspect limits must run before decoding')
    monkeypatch.setattr('demo.engine.load_image', decode_would_be_too_late)
    with pytest.raises(ValueError, match='16 megapixels'):
        read_upload(path)


def test_upload_rejects_corrupt_and_oversize_files(tmp_path):
    path = tmp_path / 'bad.png'
    path.write_text('not an image')
    with pytest.raises(ValueError, match='could not be read'):
        read_upload(path)
    with path.open('wb') as f:
        f.truncate(MAX_UPLOAD_BYTES + 1)
    with pytest.raises(ValueError, match='10 MB'):
        read_upload(path)


def test_prompt_pipeline_ranks_ascii_and_supports_multiple_requests():
    class Converter:
        chars = ' @'
        atlas = torch.stack([torch.zeros(16, 8), torch.ones(16, 8)])
        def ids(self, picture, options):
            return torch.full((1, options.cols), picture, dtype=torch.long)
    calls = []
    def generate(prompt, n, seed):
        calls.append((prompt, n, seed))
        return [0, 1]
    def score(prompt, renders):
        assert renders[0].sum() == 0 and renders[1].sum() > 0
        return torch.tensor([0.1, 0.9])
    for prompt in ['a butterfly', 'an octopus']:
        picture, text = prompt_to_ascii(Converter(), generate, score, prompt, 32, 3)
        assert picture == 1 and text == '@' * 32
    assert calls == [('a butterfly', 2, 3), ('an octopus', 2, 3)]


def test_space_bundle_excludes_local_artifacts_and_pins_runtime(tmp_path):
    bundle = build_bundle(tmp_path / 'space')
    paths = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()}
    assert {'app.py', 'LICENSE', 'packages.txt', 'demo/examples/butterfly.jpg'} <= paths
    assert not any(p.startswith(('data/', 'runs/', 'checkpoints/', '.git/')) or p == 'cat.png' for p in paths)
    assert not any(p.endswith(('.pt', '.safetensors')) for p in paths)
    assert 'torch==2.13.0' in (bundle / 'requirements.txt').read_text()
    assert 'sdk_version: 6.28.0' in (bundle / 'README.md').read_text()


def test_deployment_requests_only_zero_gpu(monkeypatch, tmp_path):
    import huggingface_hub
    from types import SimpleNamespace
    from scripts.deploy_space import deploy
    (tmp_path / 'DEPLOYMENT.json').write_text('{"uncommitted_app_changes": false}')
    calls = []
    class API:
        def create_repo(self, repo, **kwargs):
            calls.append(('create', repo, kwargs))
        def get_space_runtime(self, repo):
            return SimpleNamespace(hardware='cpu-basic', requested_hardware=None)
        def request_space_hardware(self, repo, hardware):
            calls.append(('hardware', hardware.value))
        def upload_folder(self, **kwargs):
            calls.append(('upload', kwargs['repo_type']))
            return 'published'
    monkeypatch.setattr(huggingface_hub, 'HfApi', API)
    assert deploy(tmp_path, 'SamuelReeder/asciinet') == 'published'
    assert calls[0][2]['space_hardware'] == 'zero-a10g'
    assert calls[1:] == [('hardware', 'zero-a10g'), ('upload', 'space')]


def test_deployment_refuses_uncommitted_app(tmp_path):
    from scripts.deploy_space import deploy
    (tmp_path / 'DEPLOYMENT.json').write_text('{"uncommitted_app_changes": true}')
    with pytest.raises(ValueError, match='Commit app changes'):
        deploy(tmp_path, 'SamuelReeder/asciinet')
