"""Unchanged AudioCraft processor with real trained buffers and online fitting."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
from unittest.mock import patch
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
args = parser.parse_args()
source = args.source / 'audiocraft/modules/diffusion_schedule.py'
spec = importlib.util.spec_from_file_location('original_diffusion_schedule', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(229469)
checkpoint = torch.load(args.checkpoint, mmap=True, weights_only=False, map_location='cpu')
cases = []
for band in (2, 3):
    power = float(checkpoint[band]['cfg']['processor']['power_std'])
    for trained in (True, False):
        processor = module.MultiBandProcessor(sample_rate=24000, power_std=power)
        if trained:
            processor.load_state_dict(checkpoint[band]['processor_state'], strict=True)
        processor.double()
        # Trained statistics stay exactly as stored; filters are cast, not reconstructed.
        initial_state = {name: dict(shape=list(value.shape), values=value.detach().flatten().tolist())
            for name, value in processor.state_dict().items()}
        waveform = torch.randn(2, 2, 37, requires_grad=True)
        gaussian = torch.randn_like(waveform)
        draws = []
        def draw(value):
            draws.append(1)
            return gaussian
        with patch('torch.randn_like', side_effect=draw):
            projected = processor.project_sample(waveform)
        returned = processor.return_sample(waveform)
        weights = torch.linspace(-.3, .2, waveform.numel()).reshape_as(waveform)
        loss = (projected * weights).sum() + .19 * (returned * weights.flip(-1)).sum()
        loss.backward()
        cases.append(dict(band=band, trained=trained, power=power, state=initial_state,
            waveform=waveform.detach().flatten().tolist(), gaussian=gaussian.flatten().tolist(),
            projected=projected.detach().flatten().tolist(), returned=returned.detach().flatten().tolist(),
            weights=weights.flatten().tolist(), return_weights=weights.flip(-1).flatten().tolist(),
            loss=loss.item(), gradient=waveform.grad.flatten().tolist(), draws=len(draws),
            final_count=processor.counts.item()))
destination = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/metavoice_band_processor.json'
destination.write_text(json.dumps(dict(source='https://github.com/facebookresearch/audiocraft',
    revision=subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
    release='facebook/multiband-diffusion@62daf9743d37c54e13366e396241a476908ecad7',
    unchanged_author_forwards=True, original_checkpoint=str(args.checkpoint), cases=cases)) + '\n', encoding='utf-8')
print(json.dumps(dict(cases=len(cases), draws=[x['draws'] for x in cases], counts=[x['final_count'] for x in cases])))
