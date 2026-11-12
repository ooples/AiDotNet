"""Independent CosyVoice 3 DiffRO equations 3-7, complete Qwen policy and T5 token reward.

Original Hugging Face Qwen2Model and T5ForConditionalGeneration forwards execute
unchanged. Float32-only operator requests are explicitly promoted for the native
double comparison. The configurable T5 reward is a native realization, not the
unreleased CosyVoice reward architecture/checkpoint. All objectives use summed
text log reward and summed forward token KL; only acoustic code categories are
included in the teacher-forced speech posterior.
"""
import hashlib
import inspect
import json
from pathlib import Path

import torch
import transformers
from torch import nn
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import Qwen2Config, Qwen2Model, T5Config, T5ForConditionalGeneration


class DoubleOperators(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if kwargs.get('dtype') == torch.float32:
            kwargs = dict(kwargs, dtype=torch.float64)
        elif func is torch.ops.aten.to.dtype and args[1] == torch.float32:
            args = (args[0], torch.float64, *args[2:])
        return func(*args, **kwargs)


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.manual_seed(229459)
QWEN = dict(vocab_size=32, hidden_size=8, intermediate_size=16,
    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
    rms_norm_eps=1e-6, rope_theta=1_000_000., attention_dropout=0., use_cache=False)
T5 = dict(vocab_size=11, d_model=8, d_ff=16, d_kv=4, num_heads=2,
    num_layers=1, num_decoder_layers=1, dropout_rate=0., layer_norm_epsilon=1e-6,
    decoder_start_token_id=0, eos_token_id=2, pad_token_id=0)
SPEECH, SPECIALS = 5, 200
text = torch.tensor([[4, 7, 3]])
speech = torch.tensor([[1, 3, 0, 2]])
transcript = torch.tensor([[3, 5, 2]])


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        config = Qwen2Config(**QWEN)
        config._attn_implementation = 'eager'
        self.qwen = Qwen2Model(config)
        self.speech_embedding = nn.Embedding(SPEECH + SPECIALS, 8)
        self.head = nn.Linear(8, SPEECH + SPECIALS, bias=False)

    def forward(self):
        prefix = self.speech_embedding(torch.tensor([[SPEECH]]))
        task = self.speech_embedding(torch.tensor([[SPEECH + 2]]))
        inputs = torch.cat((prefix, self.qwen.embed_tokens(text), task,
            self.speech_embedding(speech)), dim=1)
        hidden = self.qwen(inputs_embeds=inputs, use_cache=False).last_hidden_state
        return self.head(hidden)[0, text.numel() + 1:text.numel() + 1 + speech.numel(), :SPEECH]

    def named_native_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith('qwen.'):
                name = 'llm.model.model.' + name[5:]
            elif name == 'head.weight':
                name = 'llm_decoder.weight'
            yield name, parameter


class Reward(nn.Module):
    def __init__(self):
        super().__init__()
        config = T5Config(**T5)
        config._attn_implementation = 'eager'
        self.t5 = T5ForConditionalGeneration(config)
        self.speech_embedding = nn.Embedding(SPEECH, 8)

    def forward(self, probabilities):
        embedded = probabilities[None] @ self.speech_embedding.weight
        logits = self.t5(inputs_embeds=embedded, labels=transcript, use_cache=False).logits
        return F.log_softmax(logits, dim=-1).gather(-1, transcript[..., None]).sum()

    def named_native_parameters(self):
        for name, parameter in self.named_parameters():
            yield name.removeprefix('t5.'), parameter


with DoubleOperators():
    policy, reference, reward = Policy().eval(), Policy().eval(), Reward().eval()
with torch.no_grad():
    for model in (policy, reference, reward):
        for name, parameter in model.named_parameters():
            parameter.copy_(.1 * torch.randn_like(parameter) + (1 if 'norm' in name else 0))

flat = lambda value: value.detach().flatten().tolist()
weights = lambda model: {name: dict(shape=list(value.shape), values=flat(value))
    for name, value in model.named_native_parameters()}
gradients = lambda model: {name: flat(value.grad) if value.grad is not None else flat(torch.zeros_like(value))
    for name, value in model.named_native_parameters()}
result = dict(source=__doc__, torch=torch.__version__, transformers=transformers.__version__,
    qwen=QWEN, t5=T5, speech_vocabulary=SPEECH, speech_specials=SPECIALS,
    text=flat(text), speech=flat(speech), transcript=flat(transcript),
    policy_weights=weights(policy), reference_weights=weights(reference), reward_weights=weights(reward),
    source_sha256={name: hashlib.sha256(Path(inspect.getfile(model)).read_bytes()).hexdigest()
        for name, model in [('qwen', Qwen2Model), ('t5', T5ForConditionalGeneration)]}, cases=[])

for beta, temperature in [(0., .7), (.23, .7), (.23, 1.3)]:
    policy.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    reward.zero_grad(set_to_none=True)
    uniform = .05 + .9 * torch.rand(speech.numel(), SPEECH)
    with DoubleOperators():
        logits, ref_logits = policy(), reference().detach()
        logp, logref = logits.log_softmax(-1), ref_logits.log_softmax(-1)
        kl = (logp.exp() * (logp - logref)).sum()
        soft = ((logits - torch.log(-torch.log(uniform))) / temperature).softmax(-1)
        soft.retain_grad()
        log_reward = reward(soft)
        loss = -log_reward + beta * kl
    loss.backward()
    result['cases'].append(dict(beta=beta, temperature=temperature, uniforms=flat(uniform),
        policy_logits=flat(logits), reference_logits=flat(ref_logits), soft_tokens=flat(soft),
        kl=kl.item(), reward=log_reward.item(), loss=loss.item(), soft_gradient=flat(soft.grad),
        policy_gradients=gradients(policy), reward_gradients=gradients(reward),
        reference_gradients=gradients(reference)))

# ASR pretraining is real, separate SGD on the same complete reward model.
reward.zero_grad(set_to_none=True)
hard = F.one_hot(speech[0], SPEECH).to(torch.float64)
with DoubleOperators():
    before = -reward(hard)
before.backward()
rate = .001
with torch.no_grad():
    for parameter in reward.parameters():
        if parameter.grad is not None:
            parameter.add_(parameter.grad, alpha=-rate)
with DoubleOperators():
    after = -reward(hard)
result['reward_training'] = dict(learning_rate=rate, loss_before=before.item(),
    loss_after=after.item(), weights_after=weights(reward))

destination = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/cosyvoice_diffro_reference.json'
destination.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
print('DiffRO: original Qwen policy', len(result['policy_weights']), 'parameters; original T5 reward',
    len(result['reward_weights']), 'parameters; 3 complete losses/all gradients and ASR SGD.')
