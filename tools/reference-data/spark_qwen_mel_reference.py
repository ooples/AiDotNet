"""Execute Hugging Face's Qwen2 and Spark's exact torchaudio mel configuration.

The small Qwen retains grouped query attention, QKV biases, tied embeddings,
RMS epsilon and rope theta of the released Spark model. No native code is used.
"""
import json
from pathlib import Path
import torch
import torchaudio
from transformers import Qwen2Config, Qwen2ForCausalLM

torch.manual_seed(513)
torch.set_num_threads(1)
flat = lambda t: t.detach().flatten().tolist()
out = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
config = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
    num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
    rms_norm_eps=1e-6, rope_theta=1e6, tie_word_embeddings=True, attention_dropout=0.)
config._attn_implementation = 'eager'
model = Qwen2ForCausalLM(config).double().eval()
ids = torch.tensor([[3, 4, 5, 6, 7]])
data = dict(source='transformers.Qwen2ForCausalLM (released Spark configuration, reduced dimensions)',
    weights={k: flat(v) for k, v in model.state_dict().items()},
    weight_shapes={k: list(v.shape) for k, v in model.state_dict().items()},
    tokens=flat(ids), first_target=3)
logits = model(ids[:, :-1], use_cache=False).logits
loss = torch.nn.functional.cross_entropy(logits[:, 2:, :].reshape(-1, 64), ids[:, 3:].reshape(-1))
data.update(logits=flat(logits), loss=loss.item(), learning_rate=.01)
loss.backward()
with torch.no_grad():
    for p in model.parameters():
        if p.grad is not None: p.sub_(.01 * p.grad)
    data['after_sgd'] = flat(model(ids[:, :-1], use_cache=False).logits)
(out / 'spark_qwen_reference.json').write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')

transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024,
    win_length=640, hop_length=320, f_min=10., f_max=8000., n_mels=128,
    power=1., norm='slaney', mel_scale='slaney').double()
cases = []
for waveform in [torch.randn(2048, dtype=torch.float64) * .2, torch.zeros(2048, dtype=torch.float64)]:
    output = transform(waveform)
    cases.append(dict(waveform=flat(waveform), shape=list(output.shape), output=flat(output)))
(out / 'spark_mel_reference.json').write_text(json.dumps(dict(
    source='SparkAudio/Spark-TTS speaker mel configuration via torchaudio.transforms.MelSpectrogram',
    cases=cases), separators=(',', ':')), encoding='utf-8')
