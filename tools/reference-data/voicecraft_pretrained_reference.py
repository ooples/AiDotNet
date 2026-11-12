"""Original full pretrained VoiceCraft 830M Transformer references.

Arguments: jasonppy/VoiceCraft checkout, pinned original weight directory, output JSON.
All neural calculations execute unchanged author modules with original trained tensors.
Only the standard argparse.Namespace checkpoint metadata is allowlisted for weights-only loading.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import torch

checkout=Path(sys.argv[1]).resolve();weights=Path(sys.argv[2]).resolve();destination=Path(sys.argv[3]).resolve()
sys.path.insert(0,str(checkout));torch.set_num_threads(1)
from models.voicecraft import VoiceCraft
with torch.serialization.safe_globals([argparse.Namespace]):
    package=torch.load(weights/'giga830M.pth',map_location='cpu',weights_only=True,mmap=True)
args=package['config']
configuration={key:getattr(args,key) for key in ['d_model','nhead','num_decoder_layers','text_vocab_size','text_pad_token',
    'audio_vocab_size','n_codebooks','n_special','empty_token','eog','eos','audio_pad_token','max_n_spans']}
phones=package['phn2num']
# The author's constructor accepts either an int or an integer string; avoid its eval on metadata strings.
args.audio_vocab_size=int(args.audio_vocab_size)
model=VoiceCraft(args).eval();model.load_state_dict(package['model'],strict=True)
del package;gc.collect()
flat=lambda tensor:tensor.detach().flatten().tolist()
data=dict(revision='52b0f909d652ea016966804767320f13ede39abd',
    code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip(),
    config=configuration,phn2num=phones,cases=[])
for text,rows,masks in [
    ([4,5,6],[[1,2,3,4],[2049]*4,[2,3,4,5]],[-1,0,-1]),
    ([7,13,24,39,55,81],[[13,19,23,29],[2049]*4,[31,37,41,43],[2049]*4,[47,53,59,61],[67,71,73,79]],[-1,2,-1,1,-1,-1])]:
    data['cases'].append(dict(text=text,rows=rows,masks=masks))
for dtype,label in [(torch.float32,'float32'),(torch.float64,'float64'),
                    (torch.float64,'float64_position_buffers')]:
    model.to(dtype)
    if label=='float64_position_buffers':
        # SinePositionalEmbedding constructs its non-learned table explicitly in float32,
        # then merely upcasts it when model.double() runs. Preserve that original result
        # above; this additional comparison isolates table precision without changing
        # any author forward method or trained tensor.
        for embedding in [model.text_positional_embedding,model.audio_positional_embedding]:
            positions=torch.arange(embedding.pe.shape[1],dtype=dtype)[:,None]
            frequencies=torch.exp(torch.arange(0,embedding.dim_model,2,dtype=dtype)*
                                  -(torch.log(torch.tensor(10000.0,dtype=dtype))/embedding.dim_model))
            table=torch.empty(embedding.pe.shape[1],embedding.dim_model,dtype=dtype)
            table[:,0::2]=torch.sin(positions*frequencies)
            table[:,1::2]=torch.cos(positions*frequencies)
            embedding.pe=table[None].detach()
    for item in data['cases']:
        text=torch.tensor([item['text']]);rows=torch.tensor([item['rows']]);n=text.shape[1];m=rows.shape[1]
        with torch.inference_mode():
            x=model.text_positional_embedding(model.text_embedding(text))
            y=sum(model.audio_embedding[q](rows[:,:,q]) for q in range(4))
            for index,mask in enumerate(item['masks']):
                if mask>=0:y[:,index]=model.mask_embedding[mask]
            y=model.audio_positional_embedding(y)
            hidden=model.dec_forward(x,torch.tensor([n]),torch.ones(n,n,dtype=torch.bool).triu(1),torch.zeros(1,n,dtype=torch.bool),
                y,torch.tensor([m]),torch.ones(m,m,dtype=torch.bool).triu(1),torch.zeros(1,m,dtype=torch.bool))[0]
            logits=[head(hidden) for head in model.predict_layer]
        item[label]=dict(hidden=flat(hidden),logits=[flat(value) for value in logits],
            argmax=[flat(value.argmax(-1)) for value in logits])
    print('Original complete pretrained VoiceCraft 830M Transformer complete:',label,flush=True)
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print('Wrote original full VoiceCraft checkpoint references:',destination,flush=True)
