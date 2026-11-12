"""DiTTo Eq. 2: execute Hugging Face's ByT5 decoder on projected speech latents.

This checks the auxiliary mechanism, not a private DiTTo checkpoint. The miniature
uses deterministic weights and the original byte IDs, gated GELU and untied head.
"""
import json
import inspect
from pathlib import Path
import textwrap
import types
import torch
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from safetensors.torch import save_file

torch.manual_seed(2406)
torch.set_num_threads(1)
model = T5ForConditionalGeneration(T5Config(vocab_size=384, d_model=8, d_ff=16, d_kv=4,
    num_heads=2, num_layers=1, num_decoder_layers=1, dropout_rate=0, feed_forward_proj='gated-gelu',
    tie_word_embeddings=False, decoder_start_token_id=0, eos_token_id=1, pad_token_id=0)).double().eval()
for parameter in model.parameters(): parameter.requires_grad_(False)
projection = torch.nn.Linear(8, 8).double()
latents = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)
text = 'hi'
labels = torch.tensor([[value + 3 for value in text.encode('utf-8')] + [1]])
loss = model(encoder_outputs=BaseModelOutput(last_hidden_state=projection(latents)), labels=labels).loss
loss.backward()
flat = lambda tensor: tensor.detach().flatten().tolist()
state = model.state_dict()
original = dict(loss=loss.item(), latent_gradient=flat(latents.grad),
    projection_weight_gradient=flat(projection.weight.grad), projection_bias_gradient=flat(projection.bias.grad))
# HF deliberately casts RMS variance and attention softmax to float32 even for a
# double model. Also execute the SAME author methods with only those two casts
# removed, to distinguish float32 rounding from an architectural/gradient mismatch.
from transformers.models.t5.modeling_t5 import T5LayerNorm, T5Attention
for kind, before, after in [(T5LayerNorm, 'hidden_states.to(torch.float32)', 'hidden_states'),
                           (T5Attention, 'scores.float()', 'scores')]:
    method = inspect.getsource(kind.forward)
    assert before in method
    environment = dict(kind.forward.__globals__)
    exec(compile(textwrap.dedent(method).replace(before, after), '<author method, float64 accumulation>', 'exec'), environment)
    for module in model.modules():
        if isinstance(module, kind): module.forward = types.MethodType(environment['forward'], module)
projection.zero_grad(set_to_none=True)
latents.grad = None
fp64_loss = model(encoder_outputs=BaseModelOutput(last_hidden_state=projection(latents)), labels=labels).loss
fp64_loss.backward()
root = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
(root / 'ditto_semantic_reference.json').write_text(json.dumps(dict(
    source='DiTTo Eq. 2 + transformers.T5ForConditionalGeneration, miniature ByT5 configuration',
    weights={key: flat(value) for key, value in state.items()},
    weight_shapes={key: list(value.shape) for key, value in state.items()},
    projection_weight=flat(projection.weight), projection_bias=flat(projection.bias), latents=flat(latents),
    **original, fp64_loss=fp64_loss.item(), fp64_latent_gradient=flat(latents.grad),
    fp64_projection_weight_gradient=flat(projection.weight.grad), fp64_projection_bias_gradient=flat(projection.bias.grad),
    fp64_note='Same HF methods with only RMS/attention forced float32 casts removed; architecture unchanged.',
    text=text), separators=(',', ':')), encoding='utf-8')
# Float32 file import is checked separately from double-precision numerical gradients.
save_file({key: value.float().contiguous().clone() for key, value in state.items()},
    str(root / 'ditto_semantic_reference.safetensors'))
print('ByT5 decoder semantic loss', loss.item())
