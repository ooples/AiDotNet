"""Unchanged MetaVoice GPT generation, CFG, stopping, and noncausal input assembly.

Only multinomial randomness is replaced by explicit inverse-CDF uniform draws.
Neural forwards, logits filtering and generation controllers remain the author's.
"""
import argparse
import ast
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import sys
from unittest.mock import patch
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source))
from fam.llm.model import GPT, GPTConfig
from fam.llm.adapters.tilted_encodec import TiltedEncodec
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
directory = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
reference = json.loads(directory.joinpath('metavoice_gpt_reference.json').read_text())
models = {}
for stage, causal in [('first', True), ('second', False)]:
    saved = next(case for case in reference['cases'] if case['causal'] == causal and case['conditioned'] and not case['training'])
    model = GPT(GPTConfig(**saved['config']), speaker_emb_dim=3).double().eval()
    state = {name: torch.tensor(value['values']).reshape(value['shape']) for name, value in saved['parameters'].items()}
    for alias, original in saved['aliases'].items():
        state[alias] = state[original]
    model.load_state_dict(state, strict=True)
    models[stage] = (model, saved)

cases = []
variants = [(None, .8, None, True, False), (5, None, None, True, False),
    (None, .95, (3., 1.), True, False), (None, .8, (2., 2.), True, False),
    (3, None, (1., 1.), True, False), (None, 1., None, False, False),
    (None, None, None, True, True)]
model, _ = models['first']
for top_k, top_p, guidance, conditioned, terminate in variants:
    calls = []
    def multinomial(probabilities, num_samples):
        assert num_samples == 1 and probabilities.shape[0] == 1
        row = probabilities[0]
        u = (len(calls) * .1937 + .127) % 1
        if terminate and len(calls) == 1:
            assert row[16] > 0
            u = (row[:16].sum() + .5 * row[16]).item()
        token = torch.searchsorted(row.cumsum(-1), torch.tensor(u), right=True).reshape(1, 1)
        calls.append(dict(probabilities=row.tolist(), uniform=u))
        return token
    prompt = [17, 25, 19]
    speaker = torch.tensor([[[.2, -.4, .7]]]) if conditioned else None
    with patch('torch.multinomial', side_effect=multinomial):
        output = model.generate(torch.tensor([[prompt]]), 4, seq_lens=[3], temperature=.8,
            top_k=top_k, top_p=top_p, speaker_embs=speaker, batch_size=1,
            guidance_scale=guidance, dtype=torch.float64, end_of_audio_token=16, end_of_text_token=25)
    cases.append(dict(stage='first', top_k=top_k, top_p=top_p, guidance=guidance, conditioned=conditioned,
        terminate=terminate, prompt=prompt, calls=calls, output=output[0, 0].tolist()))

# Execute the original LLM.non_causal_sample body to verify its complete padding/assembly path.
source = args.source / 'fam/llm/inference.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'non_causal_sample')
scope = dict(torch=torch, Optional=Optional)
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
model, _ = models['second']
for top_k in (5, None):
    calls = []
    inputs = []
    decoded = []
    def multinomial(probabilities, num_samples):
        assert num_samples == 1
        head = len(calls)
        uniforms = [((head * 16 + row) * .17345 + .19) % 1 for row in range(16)]
        output = [torch.searchsorted(row.cumsum(-1), torch.tensor(u), right=True).item()
            for row, u in zip(probabilities, uniforms)]
        calls.append(dict(probabilities=probabilities.tolist(), uniforms=uniforms))
        return torch.tensor(output).reshape(-1, 1)
    handle = model.register_forward_pre_hook(lambda _, args: inputs.append(args[0].tolist()))
    def decode(tokens, causal):
        assert not causal
        decoded.append(tokens)
        return 'captured-original-decoder-input'
    shell = SimpleNamespace(config=SimpleNamespace(device='cpu', num_samples=1),
        tokenizer=SimpleNamespace(encode=lambda text: [16, 17]), model=model,
        _num_encodec_codebooks=8, _encodec_codes_pad_token=8, _encodec_ctx_window=16,
        speaker_cond=True, _ctx=nullcontext(), decoder=SimpleNamespace(decode=decode))
    with patch('torch.multinomial', side_effect=multinomial):
        scope['non_causal_sample'](shell, texts=['explicit-token-fixture'],
            encodec_tokens=[torch.tensor([[[0, 1, 2], [3, 4, 5]]])], batch_size=1,
            top_k=top_k, temperature=.8, speaker_embs=torch.tensor([[[.2, -.4, .7]]]))
    handle.remove()
    text, audio = TiltedEncodec(8).decode(decoded[0])
    cases.append(dict(stage='second', top_k=top_k, temperature=.8, text=[16, 17],
        first_audio=[[0, 1, 2], [3, 4, 5]], input=inputs[0][0], output=decoded[0],
        extracted_text=text, extracted_audio=audio, calls=calls))
destination = directory / 'metavoice_generation_reference.json'
destination.write_text(json.dumps(dict(source='https://github.com/metavoiceio/metavoice-src/tree/de3fa211ac4621e03a5f990651aeecc64da418f5',
    unchanged_author_forwards_and_controllers=True, inverse_cdf_multinomial_draws=True,
    models={stage: dict(config=saved['config'], parameters=saved['parameters'], aliases=saved['aliases']) for stage, (_, saved) in models.items()},
    cases=cases)) + '\n', encoding='utf-8')
print(json.dumps(dict(cases=len(cases), draws=[sum(len(call.get('uniforms', [0])) for call in case['calls']) for case in cases])))
