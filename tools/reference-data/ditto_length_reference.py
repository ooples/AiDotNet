"""Independent torch.nn.TransformerDecoder reference for DiTTo section3.2/A.7.

This checks the native disclosed causal-decoder realization, not unpublished
KRAFTON weights or unspecified private decoder hyperparameters.
"""
import json
import hashlib
import inspect
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

torch.set_default_dtype(torch.float64)
torch.manual_seed(229455)

class LengthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_input = nn.Linear(4, 8)
        self.text_input = nn.Linear(6, 8)
        self.bos = nn.Parameter(torch.zeros(1, 1, 8))
        layer = nn.TransformerDecoderLayer(8, 2, dim_feedforward=16, dropout=0,
            activation='gelu', batch_first=True, norm_first=True, layer_norm_eps=1e-5)
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.classifier = nn.Linear(8, 9)
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                parameter.uniform_(-.17, .17)
                if '.norm' in name and name.endswith('weight'):
                    parameter.add_(1)

    def forward(self, text, prompt):
        audio = self.bos if prompt is None else torch.cat([self.bos, self.audio_input(prompt)[None]], 1)
        future = torch.ones(audio.shape[1], audio.shape[1], dtype=torch.bool).triu(1)
        return self.classifier(self.decoder(audio, self.text_input(text)[None], tgt_mask=future))[0]

def flat(value):
    return value.detach().reshape(-1).tolist()

def native_parameters(model, gradient=False):
    result = {}
    def read(parameter):
        return (torch.zeros_like(parameter) if parameter.grad is None else parameter.grad) if gradient else parameter
    def linear(layer, name):
        result[name + '.weight'] = flat(read(layer.weight))
        result[name + '.bias'] = flat(read(layer.bias))
    linear(model.audio_input, 'audio_input')
    linear(model.text_input, 'text_input')
    result['bos'] = flat(read(model.bos))
    for index, block in enumerate(model.decoder.layers):
        prefix = f'blocks.{index}.'
        for attention, name in [(block.self_attn, 'self'), (block.multihead_attn, 'cross')]:
            for part, projection in enumerate(['q', 'k', 'v']):
                result[prefix + name + '.' + projection + '.weight'] = flat(read(attention.in_proj_weight)[part*8:(part+1)*8])
                result[prefix + name + '.' + projection + '.bias'] = flat(read(attention.in_proj_bias)[part*8:(part+1)*8])
            linear(attention.out_proj, prefix + name + '.out')
        for norm, name in [(block.norm1, 'self_norm'), (block.norm2, 'cross_norm'), (block.norm3, 'ff_norm')]:
            result[prefix + name + '.weight'] = flat(read(norm.weight))
            result[prefix + name + '.bias'] = flat(read(norm.bias))
        linear(block.linear1, prefix + 'ff.0')
        linear(block.linear2, prefix + 'ff.2')
    linear(model.classifier, 'classifier')
    return result

model = LengthModel().eval()
text_values = torch.randn(3, 6) * .3
prompt_values = torch.randn(3, 4) * .3
weights = native_parameters(model)
assert len(weights) == 59
cases = []
for prompt_length in [0, 3]:
    model.zero_grad(set_to_none=True)
    text = text_values.clone().requires_grad_()
    prompt = prompt_values.clone().requires_grad_() if prompt_length else None
    logits = model(text, prompt)
    labels = 7 - torch.arange(prompt_length + 1)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    values, indices = logits[-1].topk(3)
    expectation = (values.softmax(-1) * indices).sum()
    cases.append({'prompt_length': prompt_length, 'labels': labels.tolist(), 'logits': flat(logits),
        'loss': loss.item(), 'gradients': native_parameters(model, True),
        'text_gradient': flat(text.grad), 'prompt_gradient': [] if prompt is None else flat(prompt.grad),
        'expected_unrounded': expectation.item(), 'expected_rounded': expectation.round().long().item()})

cross_entropy = []
for values, labels in [([[.2, -.1, .5], [1., .1, -.2]], [2, 0]),
                       ([[1000., -1000., 0.]], [1]), ([[0., 37., -2.]], [0])]:
    logits = torch.tensor(values, requires_grad=True)
    loss = F.cross_entropy(logits, torch.tensor(labels))
    loss.backward()
    cross_entropy.append({'logits': values, 'labels': labels, 'loss': loss.item(), 'gradient': flat(logits.grad)})

boundary_posteriors = []
for selected in [0, 2, 8]:
    logits = torch.full((9,), -40.)
    logits[selected] = 40.
    values, indices = logits.topk(3)
    expectation = (values.softmax(-1) * indices).sum()
    boundary_posteriors.append({'selected': selected, 'logits': flat(logits),
        'expected_unrounded': expectation.item(), 'expected_remaining': expectation.round().long().item()})
synthesis_boundaries = [{'prompt_length': prompt, 'selected': selected,
    'expected_total': prompt + selected} for prompt, selected in [(0, 0), (3, 0), (3, 8), (9, 0), (9, 2)]]

result = {'paper': 'https://arxiv.org/html/2406.11427v2#S3.SS2',
    'scope': 'Independent torch.nn.TransformerDecoder numerical/gradient reference; native hyperparameter realization, not original private weights.',
    'torch_version': torch.__version__,
    'torch_transformer_sha256': hashlib.sha256(Path(inspect.getsourcefile(nn.TransformerDecoder)).read_bytes()).hexdigest(),
    'torch_functional_sha256': hashlib.sha256(Path(inspect.getsourcefile(F.cross_entropy)).read_bytes()).hexdigest(), 'weights': weights, 'text': flat(text_values), 'prompt': flat(prompt_values),
    'cases': cases, 'cross_entropy': cross_entropy, 'boundary_posteriors': boundary_posteriors,
    'synthesis_boundaries': synthesis_boundaries}
root = Path(__file__).resolve().parents[2]
output = root / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/ditto_length_reference.json'
output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print('Wrote two complete 59-tensor decoder cases and three loss/gradient stress cases')
