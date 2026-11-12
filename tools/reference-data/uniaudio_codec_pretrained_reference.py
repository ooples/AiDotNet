"""Full released UniAudio universal codec; unchanged author SoundStream/SEANet/RVQ.

Arguments: pinned UniAudio checkout, original weight directory, output JSON.
The 1.5 kbps TTS path selects three of the eight trained 1024-way codebooks.
"""
import argparse
import collections
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
import typing
import torch
from omegaconf import DictConfig,ListConfig,OmegaConf
from omegaconf.base import ContainerMetadata,Metadata
from omegaconf.nodes import AnyNode

checkout=Path(sys.argv[1]).resolve()
weights=Path(sys.argv[2]).resolve()
destination=Path(sys.argv[3]).resolve()
sys.path.insert(0,str(checkout/'UniAudio'))
torch.set_num_threads(1)
from tools.tokenizer.soundstream.models.soundstream import SoundStream

configuration=OmegaConf.to_container(OmegaConf.load(weights/'universal_model/config.yaml'),resolve=True)['generator']['config']
model=SoundStream(**configuration).eval()
metadata=[argparse.Namespace,DictConfig,ListConfig,ContainerMetadata,Metadata,AnyNode,
          collections.defaultdict,typing.Any,int,dict,list]
with torch.serialization.safe_globals(metadata):
    package=torch.load(weights/'universal_model/model.pth',map_location='cpu',weights_only=True,mmap=True)
model.load_state_dict(package['codec_model'],strict=True)
del package
gc.collect()
flat=lambda value:value.detach().flatten().tolist()
data=dict(revision='835f07dcf1bdf586b0f427be59e4d48d0320f00e',
          code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip(),
          config=configuration,cases=[])
for length in [1,3201,16000]:
    audio=[0.09*math.sin(i*0.037)+0.03*math.cos(i*0.011) for i in range(length)]
    data['cases'].append(dict(audio=audio))
for dtype,label in [(torch.float32,'float32'),(torch.float64,'float64')]:
    model.to(dtype)
    for item in data['cases']:
        audio=torch.tensor(item['audio'],dtype=dtype)[None,None]
        with torch.inference_mode():
            latent=model.encoder(audio)
            codes=model.encode(audio,target_bw=1.5)
            quantized=model.quantizer.decode(codes)
            waveform=model.decode(codes)
        item[label]=dict(latent=flat(latent),latent_shape=list(latent.shape),codes=flat(codes),
                         codes_shape=list(codes.shape),quantized=flat(quantized),waveform=flat(waveform))
        print('Original complete universal codec:',label,len(item['audio']),flush=True)
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print('Wrote original UniAudio codec references:',destination,flush=True)
