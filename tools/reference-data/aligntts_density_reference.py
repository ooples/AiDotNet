"""Independent PyTorch evaluation of AlignTTS sections 2.3 and 3.1.

This is a paper-equation reference, not Coqui's explicitly modified AlignTTS loss.
All strictly monotone nonempty durations are enumerated rather than using native DP.
"""
import itertools
import json
from pathlib import Path
import torch

torch.manual_seed(301950)
torch.set_num_threads(1)
model=torch.nn.Sequential(torch.nn.Linear(8,8),torch.nn.LayerNorm(8),torch.nn.ReLU(),
    torch.nn.Linear(8,8),torch.nn.LayerNorm(8),torch.nn.ReLU(),torch.nn.Linear(8,8)).double()
hidden=torch.randn(3,8,dtype=torch.float64,requires_grad=True)
mel=torch.randn(5,4,dtype=torch.float64,requires_grad=True)
output=model(hidden)
mean,log_variance=output.chunk(2,-1)
distribution=torch.distributions.Normal(mean[:,None,:],(.5*log_variance).exp()[:,None,:])
scores=distribution.log_prob(mel[None,:,:]).sum(-1)
paths=[]
for cuts in itertools.combinations(range(1,5),2):
    edges=(0,)+cuts+(5,)
    paths.append(sum(scores[phone,edges[phone]:edges[phone+1]].sum() for phone in range(3)))
loss=-torch.logsumexp(torch.stack(paths),0)
loss.backward()
flat=lambda value:value.detach().flatten().tolist()
root=Path(__file__).resolve().parents[2]/'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
(root/'aligntts_density_reference.json').write_text(json.dumps(dict(
    source='AlignTTS section 2.3 and equations 1–4, independent PyTorch Gaussian and exhaustive duration paths',
    weights={key:flat(value) for key,value in model.state_dict().items()},
    hidden=flat(hidden),mel=flat(mel),scores=flat(scores),loss=loss.item(),
    hidden_gradient=flat(hidden.grad),mel_gradient=flat(mel.grad),
    gradients={key:flat(value.grad) for key,value in model.named_parameters()}),separators=(',',':')),encoding='utf-8')
print('Enumerated AlignTTS NLL',loss.item())
