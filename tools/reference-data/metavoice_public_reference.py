"""Original actors at reduced widths with actual released vocabularies and tokenizer.

Produces original PTH fixtures, complete public synthesis token flow and the source
fine-tuning objective/AdamW update, including all frozen actor/speaker parameters.
"""
import argparse
import ast
from contextlib import nullcontext
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import sys
from unittest.mock import patch
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--release', type=Path, required=True)
parser.add_argument('--output-directory', type=Path)
parser.add_argument('--frames', type=int, default=2)
args = parser.parse_args()
sys.path.insert(0, str(args.source))
from fam.llm.model import GPT, GPTConfig
from fam.quantiser.text.tokenise import TrainedBPETokeniser
from fam.llm.adapters.flattened_encodec import FlattenedInterleavedEncodec2Codebook
from fam.llm.adapters.tilted_encodec import TiltedEncodec
from fam.llm.utils import normalize_text
from fam.llm.preprocessing.data_pipeline import pad_tokens, get_training_tuple
from fam.llm.preprocessing.audio_token_mode import combine_tokens_flattened_interleaved
speaker_module = importlib.import_module('fam.quantiser.audio.speaker_encoder.model')
speaker_module.mel_n_channels = 5
speaker_module.model_hidden_size = 8
speaker_module.model_embedding_size = 3
speaker_module.audio.mel_n_channels = 5
torch.set_num_threads(1)
torch.manual_seed(229476)
directory = args.output_directory or Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
directory.mkdir(parents=True, exist_ok=True)
if args.frames <= 0 or args.frames > 128:
    raise ValueError('Expected one to 128 independent codec frames.')
context = 16 if args.frames == 2 else 1 << (2 * args.frames + 15).bit_length()
maximum_new_tokens = 2 * args.frames + 1
release = json.loads(args.release.read_text())
metadata = torch.load(release['checkpoint_paths']['first_stage.pt'], mmap=True, weights_only=False, map_location='cpu')['meta']['tokenizer']
ar_config = GPTConfig(n_layer=2, n_head=2, n_embd=8, block_size=context, bias=False, dropout=0,
    spkemb_dropout=.1, vocab_sizes=[2562], causal=True, norm_type='rmsnorm', rmsnorm_eps=1e-5,
    nonlinearity_type='swiglu', swiglu_multiple_of=4, spk_emb_on_text=True)
nar_config = GPTConfig(n_layer=2, n_head=2, n_embd=8, block_size=context, bias=False, dropout=0,
    spkemb_dropout=0, vocab_sizes=[1538, 1025], target_vocab_sizes=[1025] * 6, causal=False,
    norm_type='layernorm', nonlinearity_type='gelu')
first = GPT(ar_config, speaker_emb_dim=3).eval()
second = GPT(nar_config, speaker_emb_dim=3).eval()
with torch.no_grad():
    for model in (first, second):
        for name, value in model.named_parameters():
            if '.ln_' in name or '.ln_f.' in name: value.uniform_(.8, 1.2)
tokenizers = []
paths = []
for name, model, config, offset in [('first', first, ar_config, 2049), ('second', second, nar_config, 1025)]:
    tokenizer_meta = dict(metadata, offset=offset)
    tokenizers.append(TrainedBPETokeniser(**tokenizer_meta))
    path = directory / ('metavoice_public_' + name + '.pt')
    torch.save(dict(model=model.state_dict(), model_args=vars(config),
        meta=dict(speaker_cond=True, speaker_emb_size=3, tokenizer=tokenizer_meta)), path)
    paths.append(path)
speaker_lstm = torch.nn.LSTM(5, 8, 3, batch_first=True)
speaker_linear = torch.nn.Linear(8, 3)
with torch.no_grad(): speaker_linear.bias.fill_(.2)
speaker_state = {'lstm.' + name: value for name, value in speaker_lstm.state_dict().items()}
speaker_state.update({'linear.' + name: value for name, value in speaker_linear.state_dict().items()})
speaker_state.update(similarity_weight=torch.tensor([10.]), similarity_bias=torch.tensor([-5.]))
speaker_path = directory / 'metavoice_public_speaker.pt'
torch.save(dict(model_state=speaker_state), speaker_path)
speaker_model = speaker_module.SpeakerEncoder(str(speaker_path), device='cpu', verbose=False, eval=True).double()
first.double(); second.double()
audio = (.1 * np.sin(np.arange(641) * 2 * np.pi * 220 / 16000)).astype(np.float32)
original_from_numpy = torch.from_numpy
with patch('torch.from_numpy', side_effect=lambda value: original_from_numpy(value).double()):
    speaker = speaker_model.embed_utterance(audio, numpy=False).reshape(1, 1, 3)
text = 'Hi…'
normalized = normalize_text(text)
uniforms = []
calls = []
desired = [token for frame in range(args.frames) for token in (frame, 1024 + frame)] + [2048]
def first_draw(probabilities, num_samples):
    row = probabilities[0]
    token = desired[len(uniforms)]
    u = (row[:token].sum() + .5 * row[token]).item()
    uniforms.append(u); calls.append(row.tolist())
    return torch.tensor([[token]])
with patch('torch.multinomial', side_effect=first_draw):
    first_output = first.generate(torch.tensor([[tokenizers[0].encode(normalized)]]), maximum_new_tokens,
        seq_lens=[len(tokenizers[0].encode(normalized))], temperature=1., top_k=None, top_p=1.,
        speaker_embs=speaker, batch_size=1, guidance_scale=(2., 2.), dtype=torch.float64,
        end_of_audio_token=2048, end_of_text_token=2561)
_, first_audio = FlattenedInterleavedEncodec2Codebook(1024).decode(first_output[0].tolist())
source = args.source / 'fam/llm/inference.py'
method = next(node for node in ast.walk(ast.parse(source.read_text(encoding='utf-8')))
    if isinstance(node, ast.FunctionDef) and node.name == 'non_causal_sample')
scope = dict(torch=torch, Optional=Optional)
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
decoded = []
def decode(tokens, causal):
    assert not causal
    decoded.append(tokens)
    return 'captured-original-decoder-input'
shell = SimpleNamespace(config=SimpleNamespace(device='cpu', num_samples=1), tokenizer=tokenizers[1],
    model=second, _num_encodec_codebooks=8, _encodec_codes_pad_token=1024, _encodec_ctx_window=context,
    speaker_cond=True, _ctx=nullcontext(), decoder=SimpleNamespace(decode=decode))
nar_uniforms = []
def second_draw(probabilities, num_samples):
    head = len(nar_uniforms) // context
    values = [((head * context + row) * .17345 + .19) % 1 for row in range(context)]
    nar_uniforms.extend(values)
    return torch.tensor([torch.searchsorted(row.cumsum(-1), torch.tensor(u), right=True).item()
        for row, u in zip(probabilities, values)]).reshape(-1, 1)
with patch('torch.multinomial', side_effect=second_draw):
    scope['non_causal_sample'](shell, texts=[normalized], encodec_tokens=[torch.tensor(first_audio)[None]],
        batch_size=1, top_k=200, temperature=1., speaker_embs=speaker)
_, final_audio = TiltedEncodec(1024).decode(decoded[0])

# Official fine-tuning: final norm and last block; text and padded EOA targets included.
for parameter in first.parameters(): parameter.requires_grad = False
for module in (first.transformer.ln_f, first.transformer.h[-1]):
    for parameter in module.parameters(): parameter.requires_grad = True
selected = [(name, value) for name, value in first.named_parameters() if value.requires_grad]
optimizer = torch.optim.AdamW([
    dict(params=[value for name, value in selected if value.dim() >= 2], weight_decay=.1),
    dict(params=[value for name, value in selected if value.dim() < 2], weight_decay=0.)],
    lr=3e-5, betas=(.9, .95), eps=1e-8)
codes = np.array([[0, 1], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
combined = combine_tokens_flattened_interleaved(codes, np.array(tokenizers[0].encode(normalized)), 1024)
padded = pad_tokens(combined, context, 2048)
x, y, se = get_training_tuple(dict(tokens=torch.tensor(padded[None], dtype=torch.long), spkemb=speaker),
    causal=True, num_codebooks=None, speaker_cond=True, device=torch.device('cpu'))
with torch.no_grad(): _, evaluation_loss = first(x, targets=y, speaker_embs=se)
first.train()
with patch('torch.rand', side_effect=lambda *shape, **kwargs: torch.full(shape, .5, device=kwargs.get('device'))):
    _, loss = first(x, targets=y, speaker_embs=se)
loss.backward()
gradient_norm = torch.nn.utils.clip_grad_norm_([value for name, value in selected], 1.)
optimizer.step()
trained = {name: dict(shape=list(value.shape), values=value.detach().flatten().tolist(), selected=value.requires_grad)
    for name, value in first.named_parameters()}
result = dict(source='https://github.com/metavoiceio/metavoice-src/tree/de3fa211ac4621e03a5f990651aeecc64da418f5',
    unchanged_author_forwards_and_controllers=True, reduced_actor_and_speaker_widths=True,
    released_vocabularies_and_tokenizer=True, checkpoint_files=[path.name for path in paths] + [speaker_path.name],
    first_config=vars(ar_config), second_config=vars(nar_config), speaker_config=dict(mels=5, hidden=8, dimension=3, layers=3),
    text=text, normalized=normalized, reference_audio=audio.tolist(), speaker=speaker.flatten().tolist(), maximum_new_tokens=maximum_new_tokens,
    uniforms=uniforms + nar_uniforms, first_output=first_output[0, 0].tolist(), final_codes=final_audio,
    training_codes=codes.tolist(), training_tokens=tokenizers[0].encode(normalized),
    evaluation_loss=evaluation_loss.item(), training_loss=loss.item(), gradient_norm=gradient_norm.item(), trained_first=trained)
destination = directory / 'metavoice_public_reference.json'
destination.write_text(json.dumps(result) + '\n', encoding='utf-8')
print(json.dumps(dict(draws=len(result['uniforms']), frames=len(final_audio[0]), selected=len(selected),
    evaluation_loss=evaluation_loss.item(), training_loss=loss.item())))
