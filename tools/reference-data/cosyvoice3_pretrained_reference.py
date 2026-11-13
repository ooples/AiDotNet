"""Original pretrained CosyVoice3 LM reference; no neural forward methods are replaced.

Arguments: FunAudioLLM/CosyVoice checkout, pinned release directory, output JSON.
Optional serving/ONNX imports are excluded by loading the unchanged class definitions.
Qwen2Encoder is attached to the real configured HF model before the complete original
CosyVoice3LM state is loaded strictly. This avoids loading an unrelated base checkpoint.
"""
import ast
import gc
import json
import logging
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import threading
from typing import Dict, Optional, Callable, List, Generator
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

checkout=Path(sys.argv[1]).resolve(); weights=Path(sys.argv[2]).resolve(); destination=Path(sys.argv[3]).resolve()
sys.path.insert(0,str(checkout)); torch.set_num_threads(1)
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import IGNORE_ID, th_accuracy
from cosyvoice.utils.mask import make_pad_mask

source=checkout/'cosyvoice/llm/llm.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
classes=[node for node in tree.body if isinstance(node,ast.ClassDef)
    and node.name in ['TransformerLM','Qwen2Encoder','Qwen2LM','CosyVoice3LM']]
online_feature=False
exec(compile(ast.fix_missing_locations(ast.Module(body=classes,type_ignores=[])),str(source),'exec'),globals())
flat=lambda value:value.detach().flatten().tolist()
config=Qwen2Config.from_pretrained(weights/'CosyVoice-BlankEN',local_files_only=True)
encoder=Qwen2Encoder.__new__(Qwen2Encoder); nn.Module.__init__(encoder)
encoder.model=Qwen2ForCausalLM(config).eval()
model=CosyVoice3LM(llm_input_size=896,llm_output_size=896,speech_token_size=6561,
    llm=encoder,sampling=None,length_normalized_loss=True,lsm_weight=0,mix_ratio=[5,15]).eval()
state=torch.load(weights/'llm.pt',map_location='cpu',weights_only=True)
model.load_state_dict(state,strict=True)
del state; gc.collect()
tokenizer_source=checkout/'cosyvoice/tokenizer/tokenizer.py'
tokenizer_classes=[node for node in ast.parse(tokenizer_source.read_text(encoding='utf-8')).body
    if isinstance(node,ast.ClassDef) and node.name in ['CosyVoice2Tokenizer','CosyVoice3Tokenizer']]
exec(compile(ast.fix_missing_locations(ast.Module(body=tokenizer_classes,type_ignores=[])),str(tokenizer_source),'exec'),globals())
tokenizer=CosyVoice3Tokenizer(weights/'CosyVoice-BlankEN')
texts=['Hello, research world!','中文与 English 123','You are a helpful assistant.<|endofprompt|>']
data=dict(revision='29e01c4e8d000f4bcd70751be16fa94bf3d85a18',
    code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip(),
    tokenizer_cases=[dict(text=text,ids=tokenizer.encode(text)) for text in texts])
text=torch.tensor([data['tokenizer_cases'][0]['ids']]); instruct=torch.tensor([data['tokenizer_cases'][2]['ids']])
speech=torch.tensor([[1,13,42,1200]])
batch=dict(text_token=text,text_token_len=torch.tensor([text.shape[1]],dtype=torch.int32),
    instruct_token=instruct,instruct_token_len=torch.tensor([instruct.shape[1]],dtype=torch.int32),
    speech_token=speech,speech_token_len=torch.tensor([speech.shape[1]],dtype=torch.int32))
data.update(text=flat(text),instruct=flat(instruct),speech=flat(speech))
for dtype,label in [(torch.float32,'float32'),(torch.float64,'float64')]:
    model.to(dtype)
    with torch.inference_mode(),patch('random.random',return_value=.9):
        result=model(batch,torch.device('cpu'))
        text_emb=model.llm.model.model.embed_tokens(text)
        instruct_emb=model.llm.model.model.embed_tokens(instruct)
        speech_emb=model.speech_embedding(speech)
        targets,inputs,lengths=model.prepare_lm_input_target(
            model.speech_embedding.weight[model.sos].reshape(1,1,-1),text,text_emb,batch['text_token_len'],
            model.speech_embedding.weight[model.task_id].reshape(1,1,-1),speech,speech_emb,batch['speech_token_len'],
            instruct,instruct_emb,batch['instruct_token_len'])
        hidden,_=model.llm(inputs,lengths)
        logits=model.llm_decoder(hidden)
    data[label]=dict(inputs=flat(inputs),targets=flat(targets),hidden=flat(hidden),logits=flat(logits),
        loss=result['loss'].item(),frames=inputs.shape[1])
    print('Original complete pretrained CosyVoice3 LM complete:',label,flush=True)
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print('Wrote original CosyVoice3 pretrained LM reference:',destination,flush=True)
