"""Execute the actual coqui/XTTS-v2 GPT, conditioner, dVAE, H/ASP and HiFi-GAN weights.

Arguments: Coqui checkout, pinned original checkpoint directory, output JSON.
Checkpoint trainer-config objects are inert metadata containers during restricted
weights-only deserialization; all neural computations use original author classes.
"""
import ast
from functools import cached_property
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types
import torch
import torchaudio
from tokenizers import Tokenizer
from num2words import num2words
from transformers import GPT2Config, GPT2PreTrainedModel, LogitsProcessorList, LogitsProcessor

checkout=Path(sys.argv[1]).resolve();weights=Path(sys.argv[2]).resolve();destination=Path(sys.argv[3]).resolve()
package=checkout/'TTS'
for name,path in [('TTS',package),('TTS.tts',package/'tts'),('TTS.tts.layers',package/'tts/layers'),
    ('TTS.tts.layers.xtts',package/'tts/layers/xtts')]:
    module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
compat=types.ModuleType('transformers.utils.model_parallel_utils')
def unavailable(*args,**kwargs): raise RuntimeError('Reference audit does not use model parallelism or implicit checkpoint downloads')
compat.get_device_map=unavailable;compat.assert_device_map=unavailable;sys.modules[compat.__name__]=compat
sys.modules['transformers'].LogitsWarper=LogitsProcessor
io=types.ModuleType('TTS.utils.io');io.load_fsspec=unavailable;sys.modules[io.__name__]=io
class CheckpointMetadata: pass
metadata_names=['TTS.tts.models.xtts.XttsAudioConfig','TTS.config.shared_configs.BaseDatasetConfig',
    'TTS.tts.models.xtts.XttsArgs','TTS.tts.configs.xtts_config.XttsConfig']
with torch.serialization.safe_globals([(CheckpointMetadata,name) for name in metadata_names]):
    root=torch.load(weights/'model.pth',map_location='cpu',weights_only=True)
state=root['model'];del root
torch.set_num_threads(1);torch.manual_seed(2294)
flat=lambda tensor:tensor.detach().flatten().tolist()
data=dict(revision='6c2b0d75eae4b7047358e3b6bd9325f857d43f77',
    code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip(),
    precision_note='GPT double reference preserves forced-float32 GroupNorm32 parameters; checkpoint metadata containers are never used as neural modules.')

# Execute the original text class without unrelated spaCy imports or package initializers.
tokenizer_source=package/'tts/layers/xtts/tokenizer.py'
tree=ast.parse(tokenizer_source.read_text(encoding='utf-8'))
namespace=dict(re=re,os=os,torch=torch,Tokenizer=Tokenizer,num2words=num2words,cached_property=cached_property,__file__=str(tokenizer_source))
nodes=[node for node in tree.body if isinstance(node,(ast.Assign,ast.FunctionDef)) or isinstance(node,ast.ClassDef) and node.name=='VoiceBpeTokenizer']
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(tokenizer_source),'exec'),namespace)
tokenizer=namespace['VoiceBpeTokenizer'](vocab_file=str(weights/'vocab.json'))
data['tokenizer_cases']=[dict(text=text,language=language,ids=tokenizer.encode(text,language)) for language,text in
    [('en','Hello, world!'),('en','Research papers matter.'),('hi','नमस्ते दुनिया 123'),('hi','"नमस्ते 123" HELLO')]]
content=data['tokenizer_cases'][0]['ids'];text=[261]+content+[0];audio=[1024,100,200,300]

from TTS.tts.layers.xtts.gpt import GPT
model=GPT(start_text_token=261,stop_text_token=0,layers=30,model_dim=1024,heads=16,
    max_text_tokens=402,max_mel_tokens=605,max_prompt_tokens=70,max_conditioning_inputs=1,
    number_text_tokens=6681,num_audio_tokens=1026,start_audio_token=1024,stop_audio_token=1025,
    checkpointing=False,use_perceiver_resampler=True).eval()
loaded=model.load_state_dict({key[4:]:value for key,value in state.items() if key.startswith('gpt.')},strict=False)
assert not loaded.missing_keys,loaded
assert all(key.endswith(('.attn.bias','.attn.masked_bias')) for key in loaded.unexpected_keys),loaded
mel=torch.randn(1,80,17)*.3
data['gpt']=dict(mel=flat(mel),frames=17,text=text,audio=audio)
for dtype,label in [(torch.float32,'float32'),(torch.float64,'float64')]:
    model.to(dtype)
    if dtype==torch.float64:
        for module in model.modules():
            if type(module).__name__=='GroupNorm32': module.float()
    with torch.inference_mode():
        condition=model.get_style_emb(mel.to(dtype)).transpose(1,2)
        ti=torch.tensor([text]);ai=torch.tensor([audio])
        te=model.text_embedding(ti)+model.text_pos_embedding(ti)
        ae=model.mel_embedding(ai)+model.mel_pos_embedding(ai)
        th,ah=model.get_logits(te,model.text_head,ae,model.mel_head,prompt=condition,return_latent=True)
        tl,al=model.get_logits(te,model.text_head,ae,model.mel_head,prompt=condition)
    data['gpt'][label]=dict(condition=flat(condition),text_hidden=flat(th),audio_hidden=flat(ah),
        text_logits=flat(tl.transpose(1,2)),audio_logits=flat(al.transpose(1,2)))
    print('Original released full GPT/conditioner complete:',label,flush=True)
decoder_latents=ah.detach().clone()
del model;gc.collect()

from TTS.tts.layers.xtts.dvae import DiscreteVAE
model=DiscreteVAE(positional_dims=1,num_tokens=1024,codebook_dim=512,num_layers=2,
    num_resnet_blocks=3,hidden_dim=512,channels=80,kernel_size=3,normalization=None,use_transposed_convs=False).double().eval()
loaded=model.load_state_dict(torch.load(weights/'dvae.pth',map_location='cpu',weights_only=True),strict=False)
assert set(loaded.missing_keys)=={'discrete_loss.accumulator_index','discrete_loss.accumulator_filled','discrete_loss.accumulator'},loaded
assert not loaded.unexpected_keys,loaded
vae_mel=torch.randn(1,80,20,dtype=torch.float64)*.3
with torch.inference_mode():
    codes=model.get_codebook_indices(vae_mel)
    reconstruction,_=model.decode(codes)
data['dvae']=dict(mel=flat(vae_mel),codes=flat(codes),reconstruction=flat(reconstruction.transpose(1,2)))
print('Original released dVAE codes and reconstruction complete',flush=True)
del model;gc.collect()

from TTS.tts.layers.xtts.hifigan_decoder import HifiDecoder
model=HifiDecoder().eval()
model.load_state_dict({key[len('hifigan_decoder.'):]:value for key,value in state.items() if key.startswith('hifigan_decoder.')},strict=True)
del state;gc.collect()
time=torch.arange(24000,dtype=torch.float32)
wave=.04*torch.randn(1,24000)+(.18*torch.sin(time*.087)+.07*torch.cos(time*.034))[None,:]
data['speaker']=dict(audio=flat(wave))
for dtype,label in [(torch.float32,'float32'),(torch.float64,'float64')]:
    model.to(dtype)
    with torch.inference_mode():
        resampled=torchaudio.functional.resample(wave.to(dtype),24000,16000)
        speaker=model.speaker_encoder(resampled,l2_norm=True)
    data['speaker'][label]=flat(speaker)
    print('Original released H/ASP waveform embedding complete:',label,flush=True)
with torch.inference_mode(): output=model(decoder_latents,g=speaker[:,:,None])
data['decoder']=dict(latents=flat(decoder_latents),frames=decoder_latents.shape[1],speaker=flat(speaker),waveform=flat(output))
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print('Original released HiFi-GAN waveform complete; wrote',destination,flush=True)
