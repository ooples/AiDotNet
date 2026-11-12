"""Real PyTorch legacy weight_norm checkpoint for the CosyVoice original-file reader."""
import json
from pathlib import Path
import torch
from safetensors.torch import save_file

torch.manual_seed(2294001)
layer = torch.nn.utils.weight_norm(torch.nn.Conv1d(2, 3, 3))
state = {'conv_pre.'+name: value.detach().contiguous() for name, value in layer.state_dict().items()}
root = Path(__file__).resolve().parents[2]/'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
torch.save(state, root/'cosyvoice_legacy_weightnorm_reference.pth')
save_file(state, root/'cosyvoice_legacy_weightnorm_reference.safetensors')
(root/'cosyvoice_legacy_weightnorm_reference.json').write_text(json.dumps(dict(
    source='torch.nn.utils.weight_norm(torch.nn.Conv1d(2,3,3)), seed 2294001',
    weights={name: value.flatten().tolist() for name, value in state.items()},
    shapes={name: list(value.shape) for name, value in state.items()})), encoding='utf-8')
print('Generated original PyTorch legacy weight-normalization checkpoints in both formats.')
