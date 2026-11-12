"""Unchanged author TextEncoder with observed original training-dropout draws."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source))
from openvoice.models import TextEncoder
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(229481)
model = TextEncoder(17, 4, 8, 16, 2, 2, 3, .1).double()
with torch.no_grad():
    for name, value in model.named_parameters():
        value.uniform_(.8, 1.2) if name.endswith('.gamma') else value.uniform_(-.2, .2)
cases = []
original = torch.nn.functional.dropout
for training, ids in [(False, [2, 5, 7, 11]), (True, [2, 5, 5, 7, 11])]:
    model.train(training); model.zero_grad(set_to_none=True)
    tokens = torch.tensor([ids]); masks = []
    def observe(value, p=.5, training=True, inplace=False):
        before = torch.random.get_rng_state()
        result = original(value, p, training, inplace)
        after = torch.random.get_rng_state()
        torch.random.set_rng_state(before)
        mask = original(torch.ones_like(value), p, training, False)
        torch.random.set_rng_state(after)
        if training:
            masks.append(dict(shape=list(mask.shape), values=mask.flatten().tolist()))
        return result
    before = torch.random.get_rng_state()
    with patch('torch.nn.functional.dropout', side_effect=observe):
        outputs = model(tokens, torch.tensor([len(ids)]))[:3]
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(before)
    with torch.no_grad(): untouched = model(tokens, torch.tensor([len(ids)]))[:3]
    assert all(torch.equal(a, b) for a, b in zip(outputs, untouched))
    torch.random.set_rng_state(after)
    assert len(masks) == (8 if training else 0)
    probes = [torch.linspace(-.31, .27, value.numel()).reshape_as(value) for value in outputs]
    loss = sum((value * probe).sum() * (index + 1) / 3 for index, (value, probe) in enumerate(zip(outputs, probes)))
    loss.backward()
    parameters = {name: dict(shape=list(value.shape), values=value.detach().flatten().tolist(), gradient=value.grad.flatten().tolist())
                  for name, value in model.named_parameters()}
    assert len(parameters) == 39
    cases.append(dict(training=training, tokens=ids, masks=masks, parameters=parameters, loss=loss.item(),
                      outputs=[value.detach().flatten().tolist() for value in outputs],
                      probes=[probe.flatten().tolist() for probe in probes]))
root = Path(__file__).resolve().parents[2]
output = root / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/openvoice_ipa_prior_reference.json'
output.write_text(json.dumps(dict(source='https://github.com/myshell-ai/OpenVoice',
    revision=subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
    unchanged_author_forward=True, dropout_observation_preserves_outputs_bitwise=True, cases=cases), separators=(',', ':')), encoding='utf-8')
print(json.dumps(dict(output=str(output), parameters=39, cases=2)))
