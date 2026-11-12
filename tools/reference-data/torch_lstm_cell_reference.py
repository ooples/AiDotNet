"""Original PyTorch LSTMCell outputs, all gradients and a real AdamW update.

The unequal nonzero input/recurrent biases must remain separate parameters.
"""
import json
from pathlib import Path
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(229483)
model = torch.nn.LSTMCell(3, 2).double()
for value in model.parameters():
    with torch.no_grad(): value.uniform_(-.3, .3)
inputs = torch.randn(4, 2, 3, requires_grad=True)
hidden = torch.randn(2, 2, requires_grad=True)
cell = torch.randn(2, 2, requires_grad=True)
h, c = hidden, cell
outputs = []
for value in inputs:
    h, c = model(value, (h, c))
    outputs.append(torch.cat((h, c), dim=1))
outputs = torch.stack(outputs)
probe = torch.linspace(-.7, .9, outputs.numel()).reshape_as(outputs)
loss = (outputs * probe).sum()
loss.backward()
weights = {name: dict(shape=list(value.shape), values=value.detach().flatten().tolist(),
                     gradient=value.grad.flatten().tolist()) for name, value in model.named_parameters()}
optimizer = torch.optim.AdamW(model.parameters(), lr=.003, betas=(.8, .99), eps=1e-9, weight_decay=.01)
optimizer.step()
for name, value in model.named_parameters(): weights[name]['updated'] = value.detach().flatten().tolist()
result = dict(torch=torch.__version__, source='torch.nn.LSTMCell',
              inputs=inputs.detach().flatten().tolist(), hidden=hidden.detach().flatten().tolist(), cell=cell.detach().flatten().tolist(),
              input_gradient=inputs.grad.flatten().tolist(), hidden_gradient=hidden.grad.flatten().tolist(), cell_gradient=cell.grad.flatten().tolist(),
              outputs=outputs.detach().flatten().tolist(), probe=probe.flatten().tolist(), loss=loss.item(), parameters=weights,
              optimizer=dict(lr=.003, betas=[.8, .99], eps=1e-9, weight_decay=.01))
path = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/torch_lstm_cell_reference.json'
path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(path)
