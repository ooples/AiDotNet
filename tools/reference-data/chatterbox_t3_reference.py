"""Execute original Chatterbox T3 conditioning, learned positions and Llama logits.

Argument: resemble-ai/chatterbox checkout. The released loss helper is also called
and its shape error recorded; the reference autoregressive objective follows the
next-token factorization documented by Resemble AI in arXiv:2605.30748, section 2.1.
"""
import json
from pathlib import Path
import sys
import types
import torch
import torch.nn.functional as F

checkout = Path(sys.argv[1]).resolve()
for name, path in [('chatterbox', checkout/'src/chatterbox'), ('chatterbox.models', checkout/'src/chatterbox/models')]:
    package = types.ModuleType(name); package.__path__ = [str(path)]; sys.modules[name] = package
from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3.modules.perceiver import Perceiver
import chatterbox.models.t3.modules.cond_enc as conditioning
conditioning.Perceiver = lambda: Perceiver(pre_attention_query_token=2, pre_attention_query_size=16,
    embedding_dim=16, num_attn_heads=2)
from chatterbox.models.t3.t3 import T3

torch.manual_seed(3520)
torch.set_num_threads(1)
LLAMA_CONFIGS['audit'] = dict(vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
    num_attention_heads=2, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
    rms_norm_eps=1e-5, rope_theta=10000, attention_dropout=0, attention_bias=False,
    mlp_bias=False, tie_word_embeddings=False, attn_implementation='eager')
hp = T3Config(text_tokens_dict_size=32)
hp.llama_config_name='audit'; hp.start_text_token=30; hp.stop_text_token=31
hp.start_speech_token=18; hp.stop_speech_token=19; hp.speech_tokens_dict_size=20
hp.max_text_tokens=16; hp.max_speech_tokens=16; hp.speaker_embed_size=6; hp.speech_cond_prompt_len=3
model = T3(hp).double().eval()
speaker = torch.randn(1, 6, dtype=torch.float64)
prompt = torch.tensor([[2, 4, 6]])
text = torch.tensor([[30, 1, 3, 31]])
speech = torch.tensor([[18, 4, 2, 5, 19]])
def condition():
    return conditioning.T3Cond(speaker_emb=speaker, cond_prompt_speech_tokens=prompt,
        emotion_adv=torch.tensor([[[.7]]], dtype=torch.float64))
arguments = dict(text_tokens=text, text_token_lens=torch.tensor([text.shape[1]]),
    speech_tokens=speech, speech_token_lens=torch.tensor([speech.shape[1]]))
output = model(t3_cond=condition(), **arguments)
cfg_embeddings, _ = model.prepare_input_embeds(t3_cond=condition(), text_tokens=text.repeat(2, 1),
    speech_tokens=speech.repeat(2, 1), cfg_weight=.5)
try:
    model.loss(t3_cond=condition(), **arguments)
    original_error = None
except RuntimeError as exception:
    original_error = str(exception)
loss = F.cross_entropy(output.text_logits[:, :-1].reshape(-1, 32), text[:, 1:].reshape(-1))
loss = loss + F.cross_entropy(output.speech_logits[:, :-1].reshape(-1, 20), speech[:, 1:].reshape(-1))
flat = lambda tensor: tensor.detach().flatten().tolist()
data = dict(source='resemble-ai/chatterbox T3.forward; next-token factorization arXiv:2605.30748 Eq. 2',
    weights={key: flat(value) for key,value in model.state_dict().items()},
    weight_shapes={key: list(value.shape) for key,value in model.state_dict().items()},
    speaker=flat(speaker), prompt=flat(prompt), text=flat(text), speech=flat(speech),
    text_logits=flat(output.text_logits), speech_logits=flat(output.speech_logits), cfg_embeddings=flat(cfg_embeddings),
    next_token_loss=loss.item(), original_training_error=original_error)
loss.backward()
with torch.no_grad():
    for parameter in model.parameters():
        if parameter.grad is not None: parameter.sub_(.01*parameter.grad)
after = model(t3_cond=condition(), **arguments)
data.update(after_sgd_text=flat(after.text_logits), after_sgd_speech=flat(after.speech_logits))
root = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
(root/'chatterbox_t3_reference.json').write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print('T3 next-token loss:', loss.item(), '; original helper:', original_error)
