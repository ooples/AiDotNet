"""Execute the four original TorToiSe inference checkpoints without miniature weights.

Arguments: neonbjb/tortoise-tts checkout, pinned HF checkpoint directory, output JSON.
The checkpoint directory contains .models/*.pth from jbetker/tortoise-tts-v2.
Only removed, unused transformers import names are supplied for compatibility;
all forward computations execute the original author classes.
"""
import gc
import json
from pathlib import Path
import subprocess
import sys
import types
import torch
import transformers
from transformers import GPT2Config, GPT2PreTrainedModel, LogitsProcessorList, LogitsProcessor

checkout=Path(sys.argv[1]).resolve(); weights=Path(sys.argv[2]).resolve(); destination=Path(sys.argv[3]).resolve()
package=types.ModuleType('tortoise');package.__path__=[str(checkout/'tortoise')];sys.modules['tortoise']=package
compat=types.ModuleType('transformers.utils.model_parallel_utils')
def unavailable(*args,**kwargs): raise RuntimeError('Checkpoint audit uses one CPU, not model parallelism')
compat.get_device_map=unavailable;compat.assert_device_map=unavailable
sys.modules[compat.__name__]=compat
sys.modules['transformers'].LogitsWarper=LogitsProcessor
torch.set_num_threads(1)
torch.manual_seed(2294)
flat=lambda tensor:tensor.detach().flatten().tolist()
load=lambda name:torch.load(weights/'.models'/name,map_location='cpu',weights_only=True)
data=dict(revision='b20a372926b4d6132bcec0a7087f5dd14c8d9e10',
    code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip(),
    precision_note='AR double variant retains float32 GroupNorm32 parameters because the original forward forcibly casts its input to float32.')

from tortoise.models.autoregressive import UnifiedVoice
model=UnifiedVoice(max_mel_tokens=604,max_text_tokens=402,max_conditioning_inputs=2,layers=30,
    model_dim=1024,heads=16,number_text_tokens=255,start_text_token=255,checkpointing=False,
    train_solo_embeddings=False).eval()
loaded=model.load_state_dict(load('autoregressive.pth'),strict=False)
assert not loaded.missing_keys,loaded
assert all(key.endswith(('.attn.bias','.attn.masked_bias')) for key in loaded.unexpected_keys),loaded
print('Original released AR checkpoint loaded; only obsolete HF causal-mask buffers ignored:',len(loaded.unexpected_keys),flush=True)
mel=torch.randn(1,80,17)*.3
text=[2,3,4,10,11];speech=[100,300,500,700]
data['ar']=dict(mel=flat(mel),frames=17,text=text,speech=speech)
for dtype,label in [(torch.float32,'float32'),(torch.float64,'float64')]:
    model.to(dtype)
    if dtype == torch.float64:
        for module in model.modules():
            if type(module).__name__ == 'GroupNorm32': module.float()
    with torch.inference_mode():
        condition=model.conditioning_encoder(mel.to(dtype))
        ti=torch.tensor([[255]+text+[0]]);si=torch.tensor([[8192]+speech])
        te=model.text_embedding(ti)+model.text_pos_embedding(ti)
        se=model.mel_embedding(si)+model.mel_pos_embedding(si)
        tl,sl=model.get_logits(condition[:,None],te,model.text_head,se,model.mel_head)
        _,latents=model.get_logits(condition[:,None],te,model.text_head,se,model.mel_head,return_latent=True)
    data['ar'][label]=dict(condition=flat(condition),text_logits=flat(tl.transpose(1,2)),
        speech_logits=flat(sl.transpose(1,2)),speech_latents=flat(latents))
    print('Original full AR forward complete:',label,flush=True)
ar_latents=torch.tensor(data['ar']['float32']['speech_latents']).reshape(1,len(speech)+1,1024)[:,:3]
del model;gc.collect()

from tortoise.models.clvp import CLVP
model=CLVP(dim_text=768,dim_speech=768,dim_latent=768,num_text_tokens=256,text_enc_depth=20,
    text_seq_len=350,text_heads=12,num_speech_tokens=8192,speech_enc_depth=20,speech_heads=12,
    speech_seq_len=430,use_xformers=True).eval()
model.load_state_dict(load('clvp2.pth'),strict=True)
data['clvp']={}
for dtype,label in [(torch.float32,'float32'),(torch.float64,'float64')]:
    model.to(dtype)
    with torch.inference_mode(): score=model(torch.tensor([text]),torch.tensor([speech]))
    data['clvp'][label]=flat(score)
    print('Original released full CLVP forward complete:',label,flush=True)
del model;gc.collect()

from tortoise.models.diffusion_decoder import DiffusionTts
model=DiffusionTts(model_channels=1024,num_layers=10,in_channels=100,out_channels=200,
    in_latent_channels=1024,in_tokens=8193,dropout=0,use_fp16=False,num_heads=16,
    layer_drop=0,unconditioned_percentage=0).eval()
model.load_state_dict(load('diffusion_decoder.pth'),strict=True)
noisy=torch.randn(1,100,9);speaker_mel=torch.randn(1,100,12)*.3;timestep=torch.tensor([137])
with torch.inference_mode():
    condition=model.get_conditioning(speaker_mel)
    aligned=model.timestep_independent(ar_latents,condition,9,False)
    prediction=model(noisy,timestep,ar_latents,condition)
    unconditional=model(noisy,timestep,ar_latents,condition,conditioning_free=True)
data['diffusion']=dict(noisy=flat(noisy),mel=flat(speaker_mel),latents=flat(ar_latents),
    condition=flat(condition),aligned=flat(aligned),prediction=flat(prediction),unconditional=flat(unconditional),timestep=137)
print('Original released full diffusion forward complete',flush=True)
del model;gc.collect()

from tortoise.models.vocoder import UnivNetGenerator
model=UnivNetGenerator()
model.eval()
model.load_state_dict(load('vocoder.pth')['model_g'],strict=True)
model.double()
mel=torch.randn(1,100,5,dtype=torch.float64)*.3;noise=torch.randn(1,64,5,dtype=torch.float64)
with torch.inference_mode(): waveform=model(mel,noise)
data['vocoder']=dict(mel=flat(mel),noise=flat(noise),waveform=flat(waveform))
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print('Original released UnivNet forward complete; wrote',destination,flush=True)
