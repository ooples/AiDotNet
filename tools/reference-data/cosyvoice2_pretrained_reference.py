"""Original complete released Qwen2LM, text tokenizer, logits and causal objective.

Arguments: original CosyVoice checkout, pinned CosyVoice2 release directory,
destination JSON. Only original class definitions are selected to exclude optional
serving imports. Neural forward and tokenizer methods are unchanged.
"""
import ast, gc, hashlib, json, logging, os, queue, random, subprocess, sys, threading
from pathlib import Path
from typing import Dict, Optional, Callable, List, Generator
from unittest.mock import patch
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from torch.utils._python_dispatch import TorchDispatchMode

class DoubleOperators(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        dtype = kwargs.get('dtype') if func is torch.ops.aten._to_copy.default else \
            args[1] if func is torch.ops.aten.to.dtype else None
        if args and isinstance(args[0], torch.Tensor) and args[0].dtype == torch.float64 and dtype == torch.float32:
            return args[0].clone()
        if dtype == torch.float32:
            # Rotary positions originate as integer IDs; advance that conversion
            # too so the unchanged author matmul receives two double operands.
            if func is torch.ops.aten._to_copy.default:
                kwargs = dict(kwargs, dtype=torch.float64)
            elif func is torch.ops.aten.to.dtype:
                args = (args[0], torch.float64, *args[2:])
        return func(*args, **kwargs)

checkout, release, destination = map(Path, sys.argv[1:4])
sys.path.insert(0, str(checkout.resolve()))
torch.set_num_threads(1)
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import IGNORE_ID, th_accuracy
from cosyvoice.utils.mask import make_pad_mask

def original_classes(path, names):
    classes = [node for node in ast.parse(path.read_text(encoding='utf-8')).body
               if isinstance(node, ast.ClassDef) and node.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])), str(path), 'exec'), globals())

online_feature = False
original_classes(checkout/'cosyvoice/llm/llm.py', ['TransformerLM', 'Qwen2Encoder', 'Qwen2LM'])
original_classes(checkout/'cosyvoice/tokenizer/tokenizer.py', ['CosyVoice2Tokenizer'])
tokenizer = CosyVoice2Tokenizer(release/'CosyVoice-BlankEN')
texts = ['Hello, research world!', '中文与 English 123', 'You are a helpful assistant.<|endofprompt|>',
         'synchronization intercontinental biodegradable', '[breath]<strong>hi</strong>[laughter]',
         '你好世界，欢迎使用语音合成。', 'trailing spaces  \n\t', 'a😀𐐀e\u0301']
flat = lambda value: value.detach().flatten().tolist()
data = dict(revision='eec1ae6c79877dbd9379285cf8789c9e0879293d',
    code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip(),
    tokenizer_vocabulary=len(tokenizer.tokenizer),
    added_tokens=tokenizer.tokenizer.get_added_vocab(),
    tokenizer_files={name: hashlib.sha256((release/'CosyVoice-BlankEN'/name).read_bytes()).hexdigest()
                     for name in ['vocab.json', 'merges.txt']},
    tokenizer_cases=[dict(text=text, ids=tokenizer.encode(text)) for text in texts])
destination.parent.mkdir(parents=True, exist_ok=True)
if '--tokenizer-only' in sys.argv:
    destination.write_text(json.dumps(data), encoding='utf-8')
    print('Original CosyVoice2 tokenizer:', data['tokenizer_vocabulary'], 'IDs', flush=True)
    raise SystemExit(0)

encoder = Qwen2Encoder.__new__(Qwen2Encoder); nn.Module.__init__(encoder)
encoder.model = Qwen2ForCausalLM(Qwen2Config.from_pretrained(release/'CosyVoice-BlankEN', local_files_only=True)).eval()
model = Qwen2LM(896, 896, 6561, llm=encoder, sampling=None,
    length_normalized_loss=True, lsm_weight=0, mix_ratio=[5, 15]).eval()
state = torch.load(release/'llm.pt', map_location='cpu', weights_only=True)
model.load_state_dict(state, strict=True)
del state; gc.collect()
text = torch.tensor([data['tokenizer_cases'][0]['ids']]); speech = torch.tensor([[1, 13, 42, 1200]])
batch = dict(text_token=text, text_token_len=torch.tensor([text.shape[1]], dtype=torch.int32),
    speech_token=speech, speech_token_len=torch.tensor([speech.shape[1]], dtype=torch.int32))
data.update(text=flat(text), speech=flat(speech))
for dtype, label in [(torch.float32, 'float32'), (torch.float64, 'float64'), (torch.float64, 'float64_operators')]:
    model.to(dtype)
    if label == 'float64_operators':
        config = json.loads((release/'CosyVoice-BlankEN/config.json').read_text())
        head = config['hidden_size']//config['num_attention_heads']
        frequency = 1/(config['rope_theta']**(torch.arange(0, head, 2, dtype=torch.float64)/head))
        rotary = model.llm.model.model.rotary_emb
        rotary.inv_freq = frequency; rotary.original_inv_freq = frequency.clone()
    with torch.inference_mode(), patch('random.random', return_value=.9), \
            (DoubleOperators() if label == 'float64_operators' else nullcontext()):
        result = model(batch, torch.device('cpu'))
        targets, inputs, lengths = model.prepare_lm_input_target(
            model.llm_embedding.weight[model.sos].reshape(1, 1, -1), text,
            model.llm.model.model.embed_tokens(text), batch['text_token_len'],
            model.llm_embedding.weight[model.task_id].reshape(1, 1, -1), speech,
            model.speech_embedding(speech), batch['speech_token_len'])
        hidden, _ = model.llm(inputs, lengths)
        logits = model.llm_decoder(hidden)
    data[label] = dict(inputs=flat(inputs), targets=flat(targets), hidden=flat(hidden), logits=flat(logits),
        loss=result['loss'].item(), frames=inputs.shape[1])
    print('Original complete pretrained CosyVoice2 LM:', label, flush=True)
destination.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
