"""Run the unchanged official Kokoro KModel with its released checkpoint and voice bank.

This is a reference generator, not a replacement implementation. Excitation draws
are captured for exact replay, while every neural forward runs unchanged. Both the
original forced-float32 signal-processing buffers and promoted-double buffers are
retained so their numerical difference cannot be mistaken for architecture parity.
"""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch
import torch
from torch.utils._python_dispatch import TorchDispatchMode

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--release', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--zero-excitation', action='store_true')
parser.add_argument('--phonemes', default='hə')
parser.add_argument('--speed', type=float, default=20.)
parser.add_argument('--excitation-seed', type=int, default=229464)
parser.add_argument('--source-fixture-output', type=Path)
args = parser.parse_args()
# Import the original model package without executing the optional Misaki pipeline import.
# This changes package discovery only; the model, modules and vocoder are untouched.
package = types.ModuleType('kokoro')
package.__path__ = [str(args.source / 'kokoro')]
sys.modules['kokoro'] = package
KModel = importlib.import_module('kokoro.model').KModel
release = json.loads(args.release.read_text(encoding='utf-8'))
files = release['files']
config = json.loads(Path(files['config.json']).read_text(encoding='utf-8'))
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(229463)
model = KModel(repo_id=release['repository'], config=config, model=files['kokoro-v1_0.pth']).eval().double()
checkpoint = torch.load(files['kokoro-v1_0.pth'], map_location='cpu', weights_only=True)
counts, identity_norm_extensions = {}, []
for name, state in checkpoint.items():
    module = getattr(model, name)
    expected = module.state_dict()
    normalized = {}
    for key, value in state.items():
        key = key.removeprefix('module.')
        if key.endswith('.weight_g'):
            key = key[:-len('.weight_g')] + '.parametrizations.weight.original0'
        elif key.endswith('.weight_v'):
            key = key[:-len('.weight_v')] + '.parametrizations.weight.original1'
        assert key not in normalized, name + '.' + key
        normalized[key] = value
    state = normalized
    assert not set(state) - set(expected), (name, set(state) - set(expected))
    for key in set(expected) - set(state):
        parent, field = key.rsplit('.', 1)
        norm = module.get_submodule(parent)
        assert isinstance(norm, torch.nn.InstanceNorm1d), (name, key)
        target = torch.ones_like(expected[key]) if field == 'weight' else torch.zeros_like(expected[key])
        assert field in ('weight', 'bias') and torch.equal(expected[key], target), (name, key)
        # Official 0.9.4 added identity affine InstanceNorm solely for legacy ONNX export.
        # Verify identity explicitly; no learned checkpoint tensor is omitted or randomized.
        identity_norm_extensions.append(name + '.' + key)
    for key, value in state.items():
        assert torch.equal(value.double(), expected[key]), name + '.' + key
    counts[name] = len(state)
voice_bank = torch.load(files['voices/af_heart.pt'], map_location='cpu', weights_only=True)
phonemes = args.phonemes
ids = [0, *[config['vocab'][symbol] for symbol in phonemes], 0]
tokens = torch.tensor([ids], dtype=torch.long)
style = voice_bank[len(phonemes) - 1].double()
captured = {}
handles = []
modules = [('bert', model.bert), ('bert_encoder', model.bert_encoder),
                     ('text_encoder', model.text_encoder), ('duration', model.predictor.duration_proj),
                     ('pitch', model.predictor.F0_proj), ('energy', model.predictor.N_proj),
                     ('decoder_encode', model.decoder.encode), ('decoder_post', model.decoder.generator.conv_post)]
modules += [('decoder_block_' + str(i), module) for i, module in enumerate(model.decoder.decode)]
modules += [('noise_conv_' + str(i), module) for i, module in enumerate(model.decoder.generator.noise_convs)]
modules += [('noise_res_' + str(i), module) for i, module in enumerate(model.decoder.generator.noise_res)]
modules += [('ups_' + str(i), module) for i, module in enumerate(model.decoder.generator.ups)]
modules += [('resblock_' + str(i), module) for i, module in enumerate(model.decoder.generator.resblocks)]
for name, module in modules:
    def capture(_, inputs, output, key=name):
        captured[key] = output.detach().flatten().tolist()
    handles.append(module.register_forward_hook(capture))
def capture_source(_, inputs, output):
    captured['harmonic_source'] = output[0].detach().flatten().tolist()
handles.append(model.decoder.generator.m_source.register_forward_hook(capture_source))
def capture_spectrum(_, inputs):
    captured['harmonic_spectrum'] = inputs[0].detach().flatten().tolist()
handles.append(model.decoder.generator.noise_convs[0].register_forward_pre_hook(capture_spectrum))

class DoubleBuffers(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if kwargs.get('dtype') == torch.float32:
            kwargs = dict(kwargs, dtype=torch.float64)
        elif func is torch.ops.aten.to.dtype and args[1] == torch.float32:
            args = (args[0], torch.float64, *args[2:])
        return func(*args, **kwargs)

def run():
    captured.clear()
    torch.manual_seed(args.excitation_seed)
    original_rand, original_normal = torch.rand, torch.randn_like
    draws = []
    def uniform(*shape, **kwargs):
        value = original_rand(*shape, **kwargs)
        if args.zero_excitation:
            value.zero_()
        draws.append(dict(kind='uniform', shape=list(value.shape), values=value.detach().flatten().tolist()))
        return value
    def normal(tensor, **kwargs):
        value = original_normal(tensor, **kwargs)
        if args.zero_excitation:
            value.zero_()
        draws.append(dict(kind='normal', shape=list(value.shape), values=value.detach().flatten().tolist()))
        return value
    with torch.no_grad(), patch('torch.rand', side_effect=uniform), patch('torch.randn_like', side_effect=normal):
        waveform, durations = model.forward_with_tokens(tokens, style, speed=args.speed)
    return dict(intermediates=dict(captured), waveform=waveform.detach().flatten().tolist(),
                durations=durations.flatten().tolist(), excitation_draws=draws)

original = run()
# TorchSTFT's window is not a registered buffer; construct it with the explicitly
# promoted dtype as well as promoting forced-float32 operator requests in the source.
for module in model.modules():
    if module.__class__.__name__ == 'TorchSTFT':
        module.window = torch.hann_window(module.win_length, periodic=True, dtype=torch.float64)
with DoubleBuffers():
    precision_matched = run()
if args.source_fixture_output:
    source_module = model.decoder.generator.m_source
    source_module.zero_grad(set_to_none=True)
    f0 = torch.tensor(precision_matched['intermediates']['pitch'], dtype=torch.float64).reshape(1, -1, 1)
    f0 = f0.repeat_interleave(300, dim=1).requires_grad_()
    recorded = precision_matched['excitation_draws']
    replay = iter(recorded)
    def next_draw(kind):
        item = next(replay)
        assert item['kind'] == kind
        return torch.tensor(item['values'], dtype=torch.float64).reshape(item['shape'])
    with DoubleBuffers(), patch('torch.rand', side_effect=lambda *shape, **kwargs: next_draw('uniform')), \
            patch('torch.randn_like', side_effect=lambda tensor, **kwargs: next_draw('normal')):
        harmonic = source_module(f0)[0]
    objective_weights = torch.linspace(-.7, .9, harmonic.numel()).reshape_as(harmonic)
    source_loss = (harmonic * objective_weights).sum()
    source_loss.backward()
    assert f0.grad is None  # The official source explicitly detaches sine/noise generation.
    source_fixture = dict(source_revision=subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256=hashlib.sha256(Path(inspect.getfile(type(source_module))).read_bytes()).hexdigest(),
        torch=torch.__version__, source='unchanged kokoro.istftnet.SourceModuleHnNSF.forward',
        pitch=precision_matched['intermediates']['pitch'], draws=recorded,
        harmonic=harmonic.detach().flatten().tolist(), objective_weights=objective_weights.flatten().tolist(),
        loss=source_loss.item(), weight=source_module.l_linear.weight.detach().flatten().tolist(),
        bias=source_module.l_linear.bias.detach().flatten().tolist(),
        weight_gradient=source_module.l_linear.weight.grad.flatten().tolist(),
        bias_gradient=source_module.l_linear.bias.grad.flatten().tolist(), pitch_detached=True)
    args.source_fixture_output.write_text(json.dumps(source_fixture) + '\n', encoding='utf-8')
for handle in handles:
    handle.remove()
result = dict(repository=release['repository'], revision=release['revision'],
    source_revision=subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
    source_sha256=hashlib.sha256(Path(inspect.getfile(KModel)).read_bytes()).hexdigest(),
    checkpoint_sha256=hashlib.sha256(Path(files['kokoro-v1_0.pth']).read_bytes()).hexdigest(),
    config=config, component_state_counts=counts, identity_norm_extensions=identity_norm_extensions,
    phonemes=phonemes, ids=ids, speed=args.speed,
    style=style.flatten().tolist(), excitation='zero draws' if args.zero_excitation else 'original Torch draws captured for native replay',
    original_forced_float32_buffers=original, precision_matched_double_buffers=precision_matched)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result))
print(json.dumps(dict(component_state_counts=counts, ids=ids,
    waveform_samples=len(original['waveform']), durations=original['durations'])), flush=True)
