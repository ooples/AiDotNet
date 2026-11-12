"""Write actual safetensors files from state dictionaries captured during upstream execution.

This preserves the author's parameter names and tensor axes. It intentionally writes
float32, the public WeightLoader's storage type, rather than a flattened native snapshot.
Run after tts_remaining_module_references.py.
"""
import json
from pathlib import Path
import torch
from safetensors.torch import save_file

root = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
for name in ['uniaudio_lm_reference', 'voicecraft_lm_reference']:
    reference = json.loads((root / (name + '.json')).read_text(encoding='utf-8'))
    state = {key: torch.tensor(value, dtype=torch.float32).reshape(reference['weight_shapes'][key])
        for key, value in reference['weights'].items()}
    save_file(state, str(root / (name + '.safetensors')),
        metadata=dict(source=reference['source'], purpose='Original named-state file import regression'))
    print(name, len(state), 'named tensors')
    if name == 'uniaudio_lm_reference':
        # Like the full release, plain tensors with Python-side metadata serialize
        # through torch._tensor._rebuild_from_type_v2. Numerical values are unchanged.
        for value in state.values():value._uniaudio_reference=True
        torch.save(dict(model=state,optimizer=dict(momentum=torch.tensor([9.0,8.0]))),
                   root/'uniaudio_wrapped_lm_reference.pth')

reference = json.loads((root / 'xtts_gpt_reference.json').read_text(encoding='utf-8'))
state = {key: torch.tensor(value, dtype=torch.float32).reshape(reference['weight_shapes'][key])
    for key, value in reference['weights'].items()}
torch.save(dict(model=state, config=dict(source=reference['source']), step=1),
    root / 'xtts_wrapped_gpt_reference.pth')
print('xtts_wrapped_gpt_reference', len(state), 'original named tensors under model root')
