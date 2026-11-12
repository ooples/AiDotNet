"""Download pinned official OpenVoice converters and run their unchanged author forwards.

Large checkpoints and reference outputs stay in --output, outside the repository.
Original stored weights are exported losslessly to safetensors for a native import audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))
from openvoice.models import SynthesizerTrn
from openvoice.mel_processing import spectrogram_torch
from openvoice import mel_processing

torch.set_num_threads(1)
releases = [
    ('v1', 'myshell-ai/OpenVoice', 'c70fc8b939bd1d8213994ff7c88e32be39708271', 'checkpoints/converter/'),
    ('v2', 'myshell-ai/OpenVoiceV2', 'f36e7edfe1684461a8343844af60babc2efbb727', 'converter/')]
for version, repo, revision, prefix in releases:
    destination = args.output / version
    destination.mkdir(parents=True, exist_ok=True)
    config_path = hf_hub_download(repo, prefix + 'config.json', revision=revision)
    checkpoint_path = hf_hub_download(repo, prefix + 'checkpoint.pth', revision=revision)
    config = json.loads(Path(config_path).read_text())
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state = checkpoint['model']
    model = SynthesizerTrn(len(config['symbols']) if 'symbols' in config else 0,
        config['data']['filter_length'] // 2 + 1,
        n_speakers=config['data']['n_speakers'], **config['model']).eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    # No silently randomized neural weights may contribute to any reported comparison.
    assert not missing and not unexpected, (version, missing, unexpected)
    save_file({name: tensor.contiguous() for name, tensor in state.items()}, str(destination / 'converter.safetensors'))
    model.double()
    torch.manual_seed(229462)
    spectrogram = torch.rand(1, 513, 9, dtype=torch.float64)
    reference = torch.rand(1, 129, 513, dtype=torch.float64)
    source = model.ref_enc(reference).unsqueeze(-1)
    target = model.ref_enc(reference.flip(1) * .7 + .2).unsqueeze(-1)
    noise = torch.randn(1, 192, 9, dtype=torch.float64)
    with torch.no_grad(), patch('torch.randn_like', side_effect=lambda x: noise):
        waveform, mask, (posterior, prior, converted) = model.voice_conversion(
            spectrogram, torch.tensor([9]), source, target, tau=.3)
        raw_audio = torch.randn(2304, dtype=torch.float64) * .05
        raw_reference = torch.randn(33024, dtype=torch.float64) * .05
        raw_spec = spectrogram_torch(raw_audio.unsqueeze(0), 1024, 22050, 256, 1024, center=False)
        ref_spec = spectrogram_torch(raw_reference.unsqueeze(0), 1024, 22050, 256, 1024, center=False)
        raw_source = model.ref_enc(raw_spec.transpose(1, 2)).unsqueeze(-1)
        raw_target = model.ref_enc(ref_spec.transpose(1, 2)).unsqueeze(-1)
        raw_waveform = model.voice_conversion(raw_spec, torch.tensor([9]), raw_source, raw_target, tau=.3)[0]
        original_window = dict(raw_source=raw_source.detach().flatten().tolist(),
            raw_target=raw_target.detach().flatten().tolist(), raw_waveform=raw_waveform.detach().flatten().tolist())
        # The author's hann_window uses the default dtype BEFORE converting to the audio dtype.
        # Preserve that float32-window result, then execute the unchanged frontend with double
        # window construction to isolate numerical buffer precision from neural architecture.
        torch.set_default_dtype(torch.float64)
        mel_processing.hann_window.clear()
        raw_spec = spectrogram_torch(raw_audio.unsqueeze(0), 1024, 22050, 256, 1024, center=False)
        ref_spec = spectrogram_torch(raw_reference.unsqueeze(0), 1024, 22050, 256, 1024, center=False)
        raw_source = model.ref_enc(raw_spec.transpose(1, 2)).unsqueeze(-1)
        raw_target = model.ref_enc(ref_spec.transpose(1, 2)).unsqueeze(-1)
        raw_waveform = model.voice_conversion(raw_spec, torch.tensor([9]), raw_source, raw_target, tau=.3)[0]
        torch.set_default_dtype(torch.float32)
        mel_processing.hann_window.clear()
    def values(x):
        return x.detach().flatten().tolist()
    result = dict(version=version, repository=repo, revision=revision,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        parameter_count=len(state), missing=missing, unexpected=unexpected,
        zero_condition=model.zero_g, config=config,
        reference=values(reference), source=values(source), target=values(target),
        spectrogram=values(spectrogram), noise=values(noise), waveform=values(waveform),
        posterior=values(posterior), prior=values(prior), converted=values(converted),
        raw_audio=values(raw_audio), raw_reference=values(raw_reference),
        raw_source=values(raw_source), raw_target=values(raw_target), raw_waveform=values(raw_waveform))
    result['original_float32_window'] = original_window
    result['raw_reference_precision'] = 'double Hann window; original float32 window outputs separately retained'
    (destination / 'reference.json').write_text(json.dumps(result))
    print(json.dumps(dict(version=version, tensors=len(state), waveform_samples=waveform.numel(), missing=missing, unexpected=unexpected)), flush=True)
