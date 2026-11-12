"""Run real Chatterbox Perceiver and CosyVoice3 DiT, including an SGD step.

Arguments: resemble-ai/chatterbox checkout, FunAudioLLM/CosyVoice checkout.
"""
import importlib.util
import json
from pathlib import Path
import sys
import torch

out = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
flat = lambda t: t.detach().flatten().tolist()
def state(model):
    return dict(weights={k: flat(v) for k, v in model.state_dict().items()},
        weight_shapes={k: list(v.shape) for k, v in model.state_dict().items()})
def update(model, loss, data):
    loss.backward()
    data['loss'] = loss.item()
    data['learning_rate'] = .01
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.sub_(data['learning_rate'] * parameter.grad)

source = Path(sys.argv[1]) / 'src/chatterbox/models/t3/modules/perceiver.py'
spec = importlib.util.spec_from_file_location('official_chatterbox_perceiver', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.manual_seed(294)
model = module.Perceiver(pre_attention_query_token=2, pre_attention_query_size=16,
    embedding_dim=16, num_attn_heads=2).double().eval()
inputs = torch.randn(1, 3, 16, dtype=torch.float64)
outputs = model(inputs)
data = dict(source='resemble-ai/chatterbox/src/chatterbox/models/t3/modules/perceiver.py',
    **state(model), input=flat(inputs), output=flat(outputs))
update(model, outputs.square().mean(), data)
data['after_sgd'] = flat(model(inputs))
(out / 'chatterbox_perceiver_reference.json').write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')

sys.path.insert(0, str(Path(sys.argv[2])))
from cosyvoice.flow.DiT.dit import DiT
torch.manual_seed(295)
model = DiT(dim=32, depth=1, heads=2, dim_head=16, ff_mult=2, mel_dim=4, mu_dim=4,
    spk_dim=4, out_channels=4, dropout=0, static_chunk_size=2, num_decoding_left_chunks=-1).double().eval()
with torch.no_grad():
    for parameter in model.parameters():
        parameter.normal_(0, .1)
x = torch.randn(1, 4, 6, dtype=torch.float64)
mu = torch.randn(1, 4, 6, dtype=torch.float64)
cond = torch.randn(1, 4, 6, dtype=torch.float64)
speaker = torch.randn(1, 4, dtype=torch.float64)
time = torch.tensor([.37], dtype=torch.float64)
def run(streaming):
    return model(x, torch.ones(1, 1, 6, dtype=torch.bool), mu, time, spks=speaker, cond=cond, streaming=streaming)
offline = run(False)
data = dict(source='FunAudioLLM/CosyVoice/cosyvoice/flow/DiT/dit.py', **state(model),
    x=flat(x.transpose(1, 2)), mu=flat(mu.transpose(1, 2)), cond=flat(cond.transpose(1, 2)),
    speaker=flat(speaker), offline=flat(offline.transpose(1, 2)), streaming=flat(run(True).transpose(1, 2)))
update(model, offline.square().mean(), data)
data['after_sgd'] = flat(run(False).transpose(1, 2))
(out / 'cosyvoice3_dit_reference.json').write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
print('Wrote Chatterbox and CosyVoice3 author-module references')
