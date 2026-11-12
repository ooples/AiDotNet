"""Original training dropout and every gradient, recording actual masks by RNG replay.

The wrappers return the unchanged original dropout/SDPA results. Replay on ones
only observes their Bernoulli masks and restores the post-call RNG state.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch
from torch.nn import functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source))
from fam.llm.model import GPT, GPTConfig
from fam.llm.preprocessing.data_pipeline import pad_tokens, get_training_tuple
from fam.llm.preprocessing.audio_token_mode import combine_tokens_flattened_interleaved
import numpy as np
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
directory = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
saved = next(case for case in json.loads(directory.joinpath('metavoice_gpt_reference.json').read_text())['cases']
    if case['causal'] and case['conditioned'] and not case['training'])
config = dict(saved['config'], dropout=.1)
model = GPT(GPTConfig(**config), speaker_emb_dim=3).double().train()
state = {name: torch.tensor(value['values']).reshape(value['shape']) for name, value in saved['parameters'].items()}
state['lm_heads.0.weight'] = state['transformer.wtes.0.weight']
model.load_state_dict(state, strict=True)
text = np.array([17, 25])
audio = np.array([[0, 1, 2], [3, 4, 5]])
combined = combine_tokens_flattened_interleaved(audio, text, 8)
padded = pad_tokens(combined, 16, 16)
tokens, targets, speaker = get_training_tuple(dict(tokens=torch.tensor(padded[None], dtype=torch.long), spkemb=torch.tensor([[[.2, -.4, .7]]])),
    causal=True, num_codebooks=None, speaker_cond=True, device=torch.device('cpu'))
cases = []
for case_index, last_blocks in enumerate((None, 1, 0)):
    freeze = last_blocks is not None
    for parameter in model.parameters():
        parameter.requires_grad = not freeze
    if freeze:
        for module in (model.transformer.ln_f, model.transformer.h[-last_blocks:]):
            for parameter in module.parameters(): parameter.requires_grad = True
    model.zero_grad(set_to_none=True)
    masks = {}
    active_dropout = []
    active_attention = []
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            label = 'embedding' if name == 'transformer.drop' else 'block.' + name.split('.')[2] + (
                '.attention_output' if 'resid_dropout' in name else '.feed_forward')
            handles += [module.register_forward_pre_hook(lambda m, a, label=label: active_dropout.append(label)),
                module.register_forward_hook(lambda m, a, o: (active_dropout.pop(), None)[1])]
        elif module.__class__.__name__ == 'SelfAttention':
            label = 'block.' + name.split('.')[2] + '.attention_probs'
            handles += [module.register_forward_pre_hook(lambda m, a, label=label: active_attention.append(label)),
                module.register_forward_hook(lambda m, a, o: (active_attention.pop(), None)[1])]
    original_dropout = F.dropout
    original_sdpa = F.scaled_dot_product_attention
    original_rand = torch.rand
    speaker_draws = []
    def record(name, shape, before, after, p):
        torch.set_rng_state(before)
        mask = original_dropout(torch.ones(shape), p, True)
        torch.set_rng_state(after)
        masks[name] = dict(shape=list(shape[1:]), values=mask.flatten().tolist())
    def dropout(value, p=.5, training=True, inplace=False):
        before = torch.get_rng_state()
        result = original_dropout(value, p, training, inplace)
        after = torch.get_rng_state()
        if training and p:
            record(active_dropout[-1], value.shape, before, after, p)
        return result
    def sdpa(query, key, value, *positional, **kwargs):
        before = torch.get_rng_state()
        result = original_sdpa(query, key, value, *positional, **kwargs)
        after = torch.get_rng_state()
        p = kwargs.get('dropout_p', 0)
        if p:
            record(active_attention[-1], (*query.shape[:-1], key.shape[-2]), before, after, p)
        return result
    def rand(*shape, **kwargs):
        result = original_rand(*shape, **kwargs)
        speaker_draws.extend(result.flatten().tolist())
        return result
    torch.manual_seed(229475 + case_index)
    with patch('torch.nn.functional.dropout', side_effect=dropout), patch('torch.nn.functional.scaled_dot_product_attention', side_effect=sdpa), patch('torch.rand', side_effect=rand):
        logits, loss = model(tokens, targets=targets, speaker_embs=speaker)
    for handle in handles: handle.remove()
    torch.manual_seed(229475 + case_index)
    with torch.no_grad():
        unobserved_logits, unobserved_loss = model(tokens, targets=targets, speaker_embs=speaker)
    assert torch.equal(unobserved_logits[0], logits[0]) and torch.equal(unobserved_loss, loss)
    loss.backward()
    parameters = {}
    for name, parameter in model.named_parameters():
        parameters[name] = dict(shape=list(parameter.shape), values=parameter.detach().flatten().tolist(),
            requires_grad=parameter.requires_grad,
            gradient=(parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).flatten().tolist())
    assert len(masks) == 7 and len(speaker_draws) == 1
    cases.append(dict(freeze=freeze, last_blocks=last_blocks, tokens=tokens[0].tolist(), targets=targets[0].tolist(),
        masks=masks, speaker_draw=speaker_draws[0], logits=logits[0].detach().flatten().tolist(), loss=loss.item(),
        parameters=parameters, selected_count=sum(value['requires_grad'] for value in parameters.values())))
destination = directory / 'metavoice_dropout_reference.json'
destination.write_text(json.dumps(dict(source='https://github.com/metavoiceio/metavoice-src/tree/de3fa211ac4621e03a5f990651aeecc64da418f5',
    unchanged_author_outputs=True, dropout_masks_observed_by_rng_replay=True, training_data_uses_original_combiner_padding_and_shift=True,
    config=config, aliases=saved['aliases'], cases=cases)) + '\n', encoding='utf-8')
print(json.dumps(dict(cases=len(cases), selected=[case['selected_count'] for case in cases], masks=7)))
