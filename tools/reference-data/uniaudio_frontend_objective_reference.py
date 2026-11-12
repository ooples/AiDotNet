"""Execute original released UniAudio text/phone tokenizers and CE/autograd.

Arguments: pinned author checkout and original weight directory.
Only unused top-level imports/demo statements are omitted. Text2PhoneTokenizer's
constructor refers to the misspelled class name TextTokenizer; a global alias
resolves that name while its constructor and tokenize method execute unchanged.
"""
import ast
import copy
import importlib.util
import json
import logging
import os
from os.path import join
from pathlib import Path
import sys
import torch

checkout=Path(sys.argv[1]).resolve()
weights=Path(sys.argv[2]).resolve()
root=Path(__file__).resolve().parents[2]/'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
torch.set_num_threads(1)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module
base=load(checkout/'UniAudio/tools/tokenizer/abs_tokenizer.py','original_uniaudio_abs').AbsTokenizer
namespace=dict(torch=torch,logging=logging,AbsTokenizer=base,join=join,copy=copy)
def original_class(path,name):
    node=next(node for node in ast.parse(path.read_text(encoding='utf-8')).body
              if isinstance(node,ast.ClassDef) and node.name==name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
    return namespace[name]
text_class=original_class(checkout/'UniAudio/tools/tokenizer/phone/text_tokenizer.py','Text2PhoneTokenizer')
namespace['TextTokenizer']=text_class
langdir=weights/'frontend-reference-data'
(langdir/'phones').mkdir(parents=True,exist_ok=True)
(langdir/'phones/optional_silence.txt').write_text('SIL\n',encoding='utf-8')
(langdir/'oov.txt').write_text('<UNK>\n',encoding='utf-8')
lexicon=langdir/'phones/align_lexicon.txt'
if not lexicon.exists():os.link(weights/'align_lexicon.txt',lexicon)
text_tokenizer=text_class(str(langdir))
phone_class=original_class(checkout/'UniAudio/tools/tokenizer/phone/phone_tokenizer.py','PhoneTokenizer')
phone_tokenizer=phone_class(str(checkout/'UniAudio/tools/tokenizer/phone/phone_dict'))
cases=[]
for text in ['hello world','Hello, world!','Hello,world','HELLO... WORLD?','[Hello] | world',
             'hello--world','hello-world','"Hello"',': ; ! ?','totallyunknownxyz','',
             '  hello\tworld\n','Dr. Smith has 123 apples.','café Straße ß','中文',
             'one one one','<UNK>','don\'t stop.']:
    phones=text_tokenizer.tokenize(text)
    ids=phone_tokenizer.tokenize(phones,task='tts').tolist()
    cases.append(dict(text=text,phones=phones,ids=ids))
phone_cases=[]
for phones in ['SIL SIL HH_B HH_B AH0_I L_I OW1_E','not_a_phone not_a_phone SIL',
               '<unk> <UNK> SIL_s SIL_m SIL_l SIL','']:
    phone_cases.append(dict(phones=phones,ids=phone_tokenizer.tokenize(phones,task='tts').tolist()))
(root/'uniaudio_frontend_reference.json').write_text(json.dumps(dict(
    source='Original Text2PhoneTokenizer.tokenize and PhoneTokenizer.tokenize(task=tts)',
    phone_inventory=phone_tokenizer.phone_dict,cases=cases,phone_cases=phone_cases),
    separators=(',',':'),ensure_ascii=False),encoding='utf-8')
model=load(checkout/'UniAudio/model.py','original_uniaudio_objective')
flat=lambda x:x.detach().flatten().tolist()
labels=torch.tensor([[2,3,4,5,0]])
mask=torch.tensor([[False,False,True,True,False]])
cases=[]
for prefix in [False,True]:
    logits=(torch.arange(35,dtype=torch.float64)*.07+.1*torch.sin(torch.arange(35,dtype=torch.float64))).reshape(1,5,7).requires_grad_()
    _,loss,_=model.CrossEntropyAndAccuracy(logits,labels,mask,prefix_lm=prefix,ignore_id=0)
    loss.backward()
    cases.append(dict(prefix_lm=prefix,logits=flat(logits),labels=flat(labels),prefix_tokens=2,
                      loss=loss.item(),gradient=flat(logits.grad)))
(root/'uniaudio_objective_reference.json').write_text(json.dumps(dict(
    source='Original UniAudio CrossEntropyAndAccuracy and PyTorch autograd',cases=cases),separators=(',',':')),encoding='utf-8')
print('Original text cases',18,'phone inventory',len(phone_tokenizer.phone_dict),'loss/gradient cases',len(cases))
