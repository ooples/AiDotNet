"""Execute the released S3Gen checkpoint, including reference preparation and rendering.

Arguments: chatterbox checkout, original weight directory, output JSON. CFM noise
and the actual stochastic vocoder excitation are exported so native comparisons
exercise the same draws rather than conflating two libraries' RNG algorithms.
"""
import gc
import json
from pathlib import Path
import subprocess
import sys
import types
import torch
from safetensors.torch import load_file

checkout=Path(sys.argv[1]).resolve();weights=Path(sys.argv[2]).resolve();destination=Path(sys.argv[3]).resolve()
for name,path in [('chatterbox',checkout/'src/chatterbox'),('chatterbox.models',checkout/'src/chatterbox/models')]:
    package=types.ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
from chatterbox.models.s3gen import S3Gen
torch.set_num_threads(1)
model=S3Gen().float().eval()
state=load_file(str(weights/'s3gen.safetensors'))
loaded=model.load_state_dict(state,strict=False)
assert set(loaded.missing_keys)<=set(model.ignore_state_dict_missing),loaded
assert not loaded.unexpected_keys,loaded
del state;gc.collect()
print('Loaded original S3Gen checkpoint',flush=True)
time=torch.arange(24000,dtype=torch.float32)
audio=(.18*torch.sin(time*.087)+.07*torch.cos(time*.034))[None,:]
with torch.inference_mode(): reference=model.embed_ref(audio,24000)
flat=lambda tensor:tensor.detach().flatten().tolist()
data=dict(source='Original ResembleAI/chatterbox S3Gen checkpoint and modules',
    revision='5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18',
    code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip(),
    audio=flat(audio),prompt_tokens=flat(reference['prompt_token']),
    prompt_mel=flat(reference['prompt_feat']),prompt_frames=reference['prompt_feat'].shape[1],
    speaker=flat(reference['embedding']))
print('Original S3Gen reference preparation complete',flush=True)
target=torch.tensor([[100,200,300,400,500,600,700,800]])
total=reference['prompt_feat'].shape[1]+2*target.shape[1]
torch.manual_seed(2294)
noise=torch.randn(1,80,total)
data.update(target_tokens=flat(target),noise=flat(noise))
for dtype,label in [(torch.float32,'float32')]:
    model.flow.to(dtype)
    arguments={key:value.to(dtype) if torch.is_tensor(value) and value.dtype.is_floating_point else value
        for key,value in reference.items()}
    with torch.inference_mode():
        mel,_=model.flow.inference(target,torch.tensor([target.shape[1]]),finalize=True,n_timesteps=10,
            noised_mels=noise.to(dtype),**arguments)
    data[label]=dict(mel=flat(mel))
    print('Original released flow complete:',label,flush=True)
model.mel2wav.float()
precision_mel=Path(str(destination)+'.mel-precision.json')
precision_cam=Path(str(destination)+'.cam-precision.json')
if precision_mel.exists() and precision_cam.exists():
    # Keep the unmodified released float32 chain above. This additional run isolates
    # neural arithmetic from the separately demonstrated frontend precision changes.
    matched=dict(reference)
    matched['prompt_feat']=torch.tensor(json.loads(precision_mel.read_text())['float64_with_float64_buffers'],dtype=torch.float32).reshape_as(reference['prompt_feat'])
    matched['embedding']=torch.tensor(json.loads(precision_cam.read_text())['embedding'],dtype=torch.float32).reshape_as(reference['embedding'])
    with torch.inference_mode():
        matched_mel,_=model.flow.inference(target,torch.tensor([target.shape[1]]),finalize=True,n_timesteps=10,noised_mels=noise,**matched)
    data['float32_with_precision_matched_conditioning']=dict(mel=flat(matched_mel))
    print('Original released flow with precision-matched reference conditioning complete',flush=True)
with torch.inference_mode():
    f0=model.mel2wav.f0_predictor(mel)
    waveform,excitation=model.mel2wav.inference(mel)
    faded=waveform.clone()
    faded[...,:len(model.trim_fade)]*=model.trim_fade.double()
data.update(f0=flat(f0),excitation=flat(excitation),waveform=flat(waveform),faded_waveform=flat(faded))
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print('Original released HiFT waveform complete; wrote',destination,flush=True)
