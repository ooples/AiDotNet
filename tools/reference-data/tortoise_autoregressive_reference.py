"""Execute neonbjb/tortoise-tts's actual UnifiedVoice; argument: upstream checkout.

Uses nonzero attention output projections and nontrivial affine norms so that missing
conditioning attention or either GPT/final normalization cannot accidentally pass.
"""
import json
import pathlib
import sys
import types
import torch

root = pathlib.Path(sys.argv[1]).resolve()
package = types.ModuleType('tortoise')
package.__path__ = [str(root / 'tortoise')]
sys.modules['tortoise'] = package
# New transformers removed these unused model-parallel helpers and renamed the
# empty sampling base class. No forward computation is substituted.
import transformers
from transformers import GPT2Config, GPT2PreTrainedModel, LogitsProcessorList, LogitsProcessor
compat = types.ModuleType('transformers.utils.model_parallel_utils')
def unavailable(*args, **kwargs):
    raise RuntimeError('Reference fixture must run on one CPU without model parallelism')
compat.get_device_map = unavailable
compat.assert_device_map = unavailable
sys.modules[compat.__name__] = compat
sys.modules['transformers'].LogitsWarper = LogitsProcessor
from tortoise.models.autoregressive import UnifiedVoice

torch.manual_seed(872)
model = UnifiedVoice(layers=1, model_dim=32, heads=4, max_text_tokens=8,
    max_mel_tokens=6, max_conditioning_inputs=2, number_text_tokens=15,
    start_text_token=15, number_mel_codes=10, start_mel_token=8,
    stop_mel_token=9, checkpointing=False).eval()
with torch.no_grad():
    for name, parameter in model.named_parameters():
        if 'conditioning_encoder.attn.' in name and name.endswith('proj_out.weight'):
            parameter.normal_(0, .03)
        if name in ('gpt.ln_f.weight', 'final_norm.weight'):
            parameter.uniform_(.7, 1.4)
        if name in ('gpt.ln_f.bias', 'final_norm.bias'):
            parameter.uniform_(-.2, .2)
    mel = torch.randn(1, 80, 7)
    condition = model.conditioning_encoder(mel)
    text = torch.tensor([[15, 2, 3, 0]])
    speech = torch.tensor([[8, 1, 4]])
    te = model.text_embedding(text) + model.text_pos_embedding(text)
    se = model.mel_embedding(speech) + model.mel_pos_embedding(speech)
    text_logits, speech_logits = model.get_logits(condition[:, None], te, model.text_head, se, model.mel_head)
    _, latents = model.get_logits(condition[:, None], te, model.text_head, se, model.mel_head, return_latent=True)
    text_loss, speech_loss, _ = model(condition, torch.tensor([[2, 3]]), torch.tensor([2]),
        torch.tensor([[1, 4]]), torch.tensor([2 * 1024]))
flat = lambda t: t.detach().flatten().tolist()
data = dict(source='neonbjb/tortoise-tts/tortoise/models/autoregressive.py UnifiedVoice',
    weights={k: flat(v) for k, v in model.state_dict().items()},
    weight_shapes={k: list(v.shape) for k, v in model.state_dict().items()},
    mel=flat(mel), condition=flat(condition), text=[2, 3], speech=[1, 4],
    text_logits=flat(text_logits.transpose(1, 2)), speech_logits=flat(speech_logits.transpose(1, 2)),
    speech_latents=flat(latents), text_loss=text_loss.item(), speech_loss=speech_loss.item())
out = pathlib.Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/tortoise_autoregressive_reference.json'
out.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
print(out)
