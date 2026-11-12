"""Original CosyVoice-300M HiFT checkpoint, F0, real excitation and waveform.

Arguments: pinned author checkout, original weights, author flow JSON, output JSON.
Original inference methods and stochastic source remain unchanged. Native waveform
decoding receives the captured original excitation, independently of RNG streams.
"""
import json, subprocess, sys
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from contextlib import nullcontext
from unittest.mock import patch

class DoubleOperators(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        dtype = kwargs.get('dtype') if func is torch.ops.aten._to_copy.default else args[1] if func is torch.ops.aten.to.dtype else None
        if dtype == torch.float32:
            if args[0].dtype == torch.float64: return args[0].clone()
            if func is torch.ops.aten._to_copy.default: kwargs = dict(kwargs, dtype=torch.float64)
            else: args = (args[0], torch.float64, *args[2:])
        return func(*args, **kwargs)

checkout, release, flow_path, destination = map(Path, sys.argv[1:5])
sys.path.insert(0, str(checkout.resolve())); torch.set_num_threads(1)
from cosyvoice.hifigan.generator import HiFTGenerator
from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor

model = HiFTGenerator(in_channels=80, base_channels=512, nb_harmonics=8, sampling_rate=22050,
    nsf_alpha=.1, nsf_sigma=.003, nsf_voiced_threshold=10, upsample_rates=[8, 8], upsample_kernel_sizes=[16, 16],
    istft_params=dict(n_fft=16, hop_len=4), resblock_kernel_sizes=[3, 7, 11], resblock_dilation_sizes=[[1, 3, 5]]*3,
    source_resblock_kernel_sizes=[7, 11], source_resblock_dilation_sizes=[[1, 3, 5]]*2,
    lrelu_slope=.1, audio_limit=.99, f0_predictor=ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)).eval()
model.load_state_dict(torch.load(release/'hift.pt', map_location='cpu', weights_only=True), strict=True)
flow = json.loads(flow_path.read_text(encoding='utf-8'))['float64_operators_derived_buffers']
mel = torch.tensor(flow['output'], dtype=torch.float64).reshape(flow['output_shape'])
flat = lambda x: x.detach().flatten().tolist()
data = dict(revision='24c40509c3c5ea6fe06b5f8790ff99e3714a6bee', code_revision=subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip(), mel=flat(mel), frames=mel.shape[2],
    double_precision='Original neural methods; double default buffers, voiced-mask casts and regenerated nonlearned periodic Hann window')
for dtype, label in [(torch.float32, 'float32'), (torch.float64, 'float64')]:
    model.to(dtype); torch.manual_seed(1335); captured = []
    if dtype == torch.float64:
        # The original SineGen allocates temporary phase buffers using the
        # default dtype. Keep these aligned with the unchanged double network.
        torch.set_default_dtype(torch.float64)
        model.stft_window = torch.hann_window(16, dtype=torch.float64)
    hook = model.f0_predictor.register_forward_hook(lambda module, args, result: captured.append(result.clone()))
    harmonics = []; phases = []; noises = []
    merge_hook = model.m_source.l_linear.register_forward_pre_hook(lambda module, args: harmonics.append(args[0].clone()))
    original_uniform = torch.distributions.Uniform.sample; original_noise = torch.randn_like
    def uniform_sample(distribution, *args, **kwargs):
        value = original_uniform(distribution, *args, **kwargs); phases.append(value.clone()); return value
    def normal_sample(*args, **kwargs):
        value = original_noise(*args, **kwargs); noises.append(value.clone()); return value
    with torch.inference_mode(), patch.object(torch.distributions.Uniform, 'sample', uniform_sample), \
            patch('torch.randn_like', normal_sample), (DoubleOperators() if dtype == torch.float64 else nullcontext()):
        waveform, source = model.inference(mel.to(dtype))
    hook.remove(); merge_hook.remove()
    data[label] = dict(f0=flat(captured[0]), source=flat(source), waveform=flat(waveform),
        harmonics=flat(harmonics[0]), harmonic_shape=list(harmonics[0].shape), phases=flat(phases[0]),
        harmonic_noise=flat(noises[0]))
    print('Original complete CosyVoice-300M HiFT:', label, flush=True)
destination.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
