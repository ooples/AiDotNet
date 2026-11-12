"""Unchanged official MetaVoice GPT forwards: causal RMS/SwiGLU and noncausal LN/GELU.

Reduced widths retain the released graph, learned absolute positions, first-stage
weight tying, second-stage six heads, speaker conditioning, ignored text targets,
the mean-over-heads cross entropy and every parameter/input gradient.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))
from fam.llm.model import GPT, GPTConfig

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
cases = []
for causal in (True, False):
    config = GPTConfig(block_size=16, vocab_sizes=[26] if causal else [18, 9],
        target_vocab_sizes=None if causal else [9] * 6, n_layer=2, n_head=2, n_embd=8,
        dropout=0., spkemb_dropout=.1 if causal else 0., bias=False, causal=causal, spk_emb_on_text=True,
        norm_type='rmsnorm' if causal else 'layernorm', rmsnorm_eps=1e-5 if causal else None,
        nonlinearity_type='swiglu' if causal else 'gelu', swiglu_multiple_of=4 if causal else None)
    torch.manual_seed(229463 + int(causal))
    model = GPT(config, speaker_emb_dim=3).double().eval()
    with torch.no_grad():
        for name, value in model.named_parameters():
            value.uniform_(.8, 1.2) if '.ln_' in name or '.ln_f.' in name else value.uniform_(-.08, .08)
    tokens = torch.tensor([[[17, 25, 1, 10]]] if causal else [[[9, 17, 1, 2], [8, 8, 4, 5]]])
    targets = torch.tensor([[[-1, 3, 5, 4]]] if causal else [[[-1, -1, i % 8, (i + 1) % 8] for i in range(6)]])
    variants = [(True, False, None), (False, False, None)]
    if causal:
        variants += [(True, True, .05), (True, True, .5)]
    for conditioned, training, dropout_draw in variants:
        model.train(training)
        model.zero_grad(set_to_none=True)
        speaker = torch.tensor([[[.2, -.4, .7]]], requires_grad=True)
        capture = {}
        handle = model.transformer.ln_f.register_forward_hook(lambda _, inputs, output: capture.update(hidden=output))
        if training:
            with patch('torch.rand', side_effect=lambda *shape, **kwargs: torch.full(shape, dropout_draw,
                    device=kwargs.get('device'), dtype=torch.float64)):
                logits, cross_entropy = model(tokens, targets=targets, speaker_embs=speaker if conditioned else None)
        else:
            logits, cross_entropy = model(tokens, targets=targets, speaker_embs=speaker if conditioned else None)
        handle.remove()
        hidden = capture['hidden']
        hidden_weights = torch.linspace(-.2, .3, hidden.numel()).reshape_as(hidden)
        loss = cross_entropy + .07 * (hidden * hidden_weights).sum()
        loss.backward()
        parameters = {}
        for name, value in model.named_parameters():
            gradient = value.grad if value.grad is not None else torch.zeros_like(value)
            parameters[name] = dict(shape=list(value.shape), values=value.detach().flatten().tolist(),
                gradient=gradient.flatten().tolist(), gradient_is_none=value.grad is None)
        case = dict(causal=causal, conditioned=conditioned, training=training, dropout_draw=dropout_draw, config=vars(config),
            tokens=tokens.tolist(), targets=targets.tolist(), speaker=speaker.detach().flatten().tolist(),
            hidden=hidden.detach().flatten().tolist(), hidden_weights=hidden_weights.flatten().tolist(),
            logits=[value.detach().flatten().tolist() for value in logits], cross_entropy=cross_entropy.item(),
            loss=loss.item(), speaker_gradient=(speaker.grad if speaker.grad is not None else torch.zeros_like(speaker)).flatten().tolist(),
            parameters=parameters, unique_parameter_count=len(parameters),
            aliases={'lm_heads.0.weight': 'transformer.wtes.0.weight'} if causal else {})
        model.eval()
        with torch.no_grad():
            inference_logits, _ = model(tokens, speaker_embs=speaker if conditioned else None)
        case['inference_logits'] = [value.flatten().tolist() for value in inference_logits]
        cases.append(case)
source_file = Path(inspect.getfile(GPT))
result = dict(source='https://github.com/metavoiceio/metavoice-src',
    revision=subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
    source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(), torch=torch.__version__,
    unchanged_author_forwards=True, reduced_width=True, cases=cases)
destination = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/metavoice_gpt_reference.json'
destination.write_text(json.dumps(result) + '\n', encoding='utf-8')
print(json.dumps(dict(cases=len(cases), unique_parameters=[case['unique_parameter_count'] for case in cases])))
