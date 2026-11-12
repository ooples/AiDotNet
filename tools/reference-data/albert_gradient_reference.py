"""Unchanged Hugging Face ALBERT: shared blocks, repeated IDs and a nonzero loaded padding row."""
import json
from pathlib import Path
import torch
import transformers
from transformers import AlbertConfig, AlbertModel

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(229485)
config = AlbertConfig(vocab_size=17, embedding_size=4, hidden_size=8, num_hidden_layers=3,
                      num_hidden_groups=1, inner_group_num=1, num_attention_heads=2,
                      intermediate_size=16, max_position_embeddings=16, type_vocab_size=2,
                      hidden_dropout_prob=0., attention_probs_dropout_prob=0.,
                      hidden_act='gelu_new', layer_norm_eps=1e-12, pad_token_id=0)
model = AlbertModel(config).double().eval()
with torch.no_grad():
    for name, value in model.named_parameters():
        value.uniform_(.7, 1.1) if name.endswith('LayerNorm.weight') or name.endswith('layer_norm.weight') else value.uniform_(-.2, .2)
assert model.embeddings.word_embeddings.weight[0].abs().sum() > 0
ids = torch.tensor([[0, 3, 3, 7, 0]])
output = model(ids, attention_mask=torch.ones_like(ids)).last_hidden_state
probe = torch.linspace(-.7, .9, output.numel()).reshape_as(output)
loss = (output*probe).sum()
loss.backward()
parameters = {name: dict(shape=list(value.shape), values=value.detach().flatten().tolist(),
                         gradient=value.grad.flatten().tolist())
              for name,value in model.named_parameters() if not name.startswith('pooler.')}
assert len(parameters) == 23
assert torch.count_nonzero(model.embeddings.word_embeddings.weight.grad[0]) == 0
result = dict(torch=torch.__version__, transformers=transformers.__version__, config=config.to_dict(),
              ids=ids.flatten().tolist(), output=output.detach().flatten().tolist(),
              probe=probe.flatten().tolist(), loss=loss.item(), parameters=parameters)
path=Path(__file__).resolve().parents[2]/'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/albert_gradient_reference.json'
path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(path)
