"""Original complete CosyVoice-300M TransformerLM, embeddings, logits and loss.

Arguments: author checkout, original release directory, output JSON. Original
neural methods are unchanged. Original float32 and double executions are retained;
a separate double reference regenerates only nonlearned sinusoidal tables.
"""
import ast, gc, json, logging, math, os, queue, random, subprocess, sys, threading, types
from pathlib import Path
from typing import Dict, Optional, Callable, List, Generator
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from transformers import AutoModelForCausalLM

checkout, release, destination = map(Path, sys.argv[1:4])
sys.path.insert(0, str(checkout.resolve())); torch.set_num_threads(1)
# The encoder's registry also imports optional serving models. Retain its original
# activation/subsampling/position/attention dictionaries without those imports.
registry_path = checkout/'cosyvoice/utils/class_utils.py'
registry_source = '\n'.join(line for line in registry_path.read_text(encoding='utf-8').splitlines()
    if not line.startswith(('from cosyvoice.llm', 'from cosyvoice.flow', 'from cosyvoice.hifigan', 'from cosyvoice.cli')))
registry = types.ModuleType('cosyvoice.utils.class_utils')
sys.modules[registry.__name__] = registry
exec(compile(registry_source.split('def get_model_type')[0], str(registry_path), 'exec'), registry.__dict__)
from cosyvoice.transformer.encoder import ConformerEncoder, TransformerEncoder
from cosyvoice.transformer.embedding import PositionalEncoding, EspnetRelPositionalEncoding
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import IGNORE_ID, th_accuracy
from cosyvoice.utils.mask import make_pad_mask

source = checkout/'cosyvoice/llm/llm.py'
online_feature = False
classes = [node for node in ast.parse(source.read_text(encoding='utf-8')).body
           if isinstance(node, ast.ClassDef) and node.name == 'TransformerLM']
exec(compile(ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])), str(source), 'exec'), globals())
text_encoder = ConformerEncoder(512, output_size=1024, attention_heads=16, linear_units=4096, num_blocks=6,
    dropout_rate=.1, positional_dropout_rate=.1, attention_dropout_rate=0, normalize_before=True,
    input_layer='linear', pos_enc_layer_type='rel_pos_espnet', selfattention_layer_type='rel_selfattn',
    use_cnn_module=False, macaron_style=False, use_dynamic_chunk=False, use_dynamic_left_chunk=False, static_chunk_size=1)
decoder = TransformerEncoder(1024, output_size=1024, attention_heads=16, linear_units=4096, num_blocks=14,
    dropout_rate=.1, positional_dropout_rate=.1, attention_dropout_rate=0, input_layer='linear_legacy',
    pos_enc_layer_type='rel_pos_espnet', selfattention_layer_type='rel_selfattn', static_chunk_size=1)
model = TransformerLM(512, 1024, 1024, 51866, 4096, text_encoder=text_encoder, llm=decoder,
    spk_embed_dim=192, sampling=None, length_normalized_loss=True, lsm_weight=0).eval()
model.load_state_dict(torch.load(release/'llm.pt', map_location='cpu', weights_only=True), strict=True)
print('Loaded all original CosyVoice-300M TransformerLM tensors.', flush=True)
flat = lambda value: value.detach().flatten().tolist()
text = torch.tensor([[15947, 11, 2057, 1002, 0]])
speech = torch.tensor([[1, 13, 42, 1200]])
data = dict(revision='24c40509c3c5ea6fe06b5f8790ff99e3714a6bee', code_revision=subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip(), text=flat(text), speech=flat(speech))

def execute(dtype):
    vector = (torch.cos(torch.arange(192, dtype=dtype)*.13)*.17)[None]
    batch = dict(text_token=text, text_token_len=torch.tensor([text.shape[1]]), speech_token=speech,
        speech_token_len=torch.tensor([speech.shape[1]]), embedding=vector)
    with torch.inference_mode():
        loss = model(batch, torch.device('cpu'))['loss']
        encoded, _ = model.encode(model.text_embedding(text), batch['text_token_len'])
        speaker = model.spk_embed_affine_layer(F.normalize(vector, dim=1)).unsqueeze(1)
        sos_id = model.sos if hasattr(model, 'sos') else model.sos_eos
        sos = model.llm_embedding.weight[sos_id].reshape(1, 1, -1)
        task = model.llm_embedding.weight[model.task_id].reshape(1, 1, -1)
        inputs = torch.cat([sos, speaker, encoded, task, model.speech_embedding(speech)], dim=1)
        hidden, _ = model.llm(inputs, torch.tensor([inputs.shape[1]]))
        logits = model.llm_decoder(hidden)
    return dict(xvector=flat(vector), encoded_text=flat(encoded), inputs=flat(inputs), hidden=flat(hidden),
        logits=flat(logits), loss=loss.item(), frames=inputs.shape[1])

data['original_float32'] = execute(torch.float32)
model.double(); gc.collect()
data['original_float64_float32_positions'] = execute(torch.float64)
for module in model.modules():
    if isinstance(module, (PositionalEncoding, EspnetRelPositionalEncoding)):
        if isinstance(module, EspnetRelPositionalEncoding):
            half = (module.pe.shape[1]-1)//2
            position = torch.arange(half, -half-1, -1, dtype=torch.float64)[:, None]
        else:
            position = torch.arange(module.max_len, dtype=torch.float64)[:, None]
        frequency = torch.exp(torch.arange(0, module.d_model, 2, dtype=torch.float64)*-(math.log(10000.0)/module.d_model))
        pe = torch.empty(position.shape[0], module.d_model, dtype=torch.float64)
        pe[:, 0::2] = torch.sin(position*frequency); pe[:, 1::2] = torch.cos(position*frequency)
        module.pe = pe[None]
data['float64_derived_positions'] = execute(torch.float64)
destination.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
print('Generated complete original CosyVoice-300M LM references at all three precisions.', flush=True)
