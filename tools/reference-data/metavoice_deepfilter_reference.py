"""Independent DeepFilterNet3 author-graph fixtures for MetaVoice's default enhancer.

The original neural forwards are not replaced. libdf supplies the published ERB
bank; reduced graphs exercise every active parameter, including separate GRU
biases. Full checkpoint and waveform evidence is emitted separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--dependencies', type=Path, required=True)
parser.add_argument('--model-directory', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--reduced-output', type=Path)
parser.add_argument('--frontend-output', type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.dependencies))
sys.path.insert(0, str(args.source / 'DeepFilterNet'))
import numpy as np
import torch
from df.config import config
from df.enhance import init_df, df_features, enhance
from df.deepfilternet3 import init_model
from libdf import DF

torch.set_num_threads(1)
torch.manual_seed(229473)
model, state, _ = init_df(str(args.model_directory))
checkpoint = args.model_directory / 'checkpoints' / 'model_120.ckpt.best'


def tensor(value):
    return dict(shape=list(value.shape), values=value.detach().flatten().tolist())


def run_case(net, batch, frames, frequencies, df_bins, erb_bins, include_parameters):
    net.zero_grad(set_to_none=True)
    dtype = next(net.parameters()).dtype
    spec = torch.randn(batch, 1, frames, frequencies, 2, dtype=dtype, requires_grad=True)
    erb = torch.randn(batch, 1, frames, erb_bins, dtype=dtype, requires_grad=True)
    features = torch.randn(batch, 1, frames, df_bins, 2, dtype=dtype, requires_grad=True)
    outputs = net(spec, erb, features)
    weights = [torch.linspace(-.23, .31, out.numel(), dtype=dtype).reshape_as(out)
               for out in outputs]
    loss = sum((out * weight).sum() for out, weight in zip(outputs, weights))
    loss.backward()
    parameters = {}
    for name, value in net.named_parameters():
        item = dict(shape=list(value.shape), gradient=None if value.grad is None
                    else value.grad.flatten().tolist())
        if include_parameters:
            item['values'] = value.detach().flatten().tolist()
        parameters[name] = item
    unused = [name for name, item in parameters.items() if item['gradient'] is None]
    # The author constructs df_fc_a but never calls it in DfDecoder.forward.
    assert unused == ['df_dec.df_fc_a.0.weight', 'df_dec.df_fc_a.0.bias'], unused
    return dict(spec=tensor(spec), erb=tensor(erb), features=tensor(features),
                outputs=[tensor(out) for out in outputs], weights=[tensor(w) for w in weights],
                loss=loss.item(), input_gradients=[tensor(x.grad) for x in (spec, erb, features)],
                parameters=parameters,
                buffers={name: tensor(value) for name, value in net.named_buffers()})


# Original trained float32 waveform and frontend: retain the backend's real DSP.
audio = (torch.sin(torch.arange(3840) * .071) * .13
         + torch.cos(torch.arange(3840) * .019) * .04).unsqueeze(0)
spectrogram, erb_feature, spec_feature = df_features(audio, state, 96, device='cpu')
original_synthesis = torch.from_numpy(state.synthesis(
    torch.view_as_complex(spectrogram.squeeze(1)).numpy().copy()))
state.reset()
waveform = enhance(model, state, audio)
full_frontend = dict(audio=tensor(audio), spec=tensor(spectrogram),
                     erb=tensor(erb_feature), features=tensor(spec_feature),
                     waveform=tensor(waveform), synthesis=tensor(original_synthesis),
                     window=tensor(torch.from_numpy(state.fft_window())),
                     erb_widths=state.erb_widths().tolist())
model.double()
full = run_case(model, 1, 7, 481, 96, 32, False)
with torch.no_grad():
    padded = torch.nn.functional.pad(audio, (0, 960))
    inputs = df_features(padded, state, 96, device='cpu')
    enhanced = model(*(x.double() for x in inputs))[0].float()
    matched = torch.from_numpy(state.synthesis(torch.view_as_complex(enhanced.squeeze(1)).numpy().copy()))
    full_frontend['matched_waveform'] = tensor(matched[:, 480:audio.shape[-1] + 480])

# Change only constructor configuration, not the author's implementation.
for section, key, value in [('df', 'fft_size', 64), ('df', 'hop_size', 32),
                            ('df', 'nb_erb', 8), ('df', 'nb_df', 8),
                            ('deepfilternet', 'conv_ch', 4),
                            ('deepfilternet', 'emb_hidden_dim', 8),
                            ('deepfilternet', 'df_hidden_dim', 8),
                            ('deepfilternet', 'linear_groups', 2),
                            ('deepfilternet', 'enc_linear_groups', 2)]:
    config.set(key, value, int, section=section)
reduced_state = DF(sr=48000, fft_size=64, hop_size=32, nb_bands=8)
reduced = init_model(reduced_state).double().eval()
cases = [run_case(reduced, batch, frames, 33, 8, 8, True)
         for batch, frames in [(1, 7), (2, 9)]]
result = dict(source_revision=subprocess.check_output(
    ['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
    checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    full=full, full_frontend=full_frontend, cases=cases)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
if args.reduced_output:
    args.reduced_output.parent.mkdir(parents=True, exist_ok=True)
    args.reduced_output.write_text(json.dumps(dict(source_revision=result['source_revision'],
        checkpoint_sha256=result['checkpoint_sha256'], cases=cases), separators=(',', ':')), encoding='utf-8')
if args.frontend_output:
    args.frontend_output.parent.mkdir(parents=True, exist_ok=True)
    args.frontend_output.write_text(json.dumps(dict(source_revision=result['source_revision'],
        frontend=full_frontend), separators=(',', ':')), encoding='utf-8')
print(json.dumps(dict(output=str(args.output), tensors=len(model.state_dict()),
                      parameters=len(list(model.parameters())), cases=len(cases))))
