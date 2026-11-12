"""Execute the real released Chatterbox voice encoder and 520M T3 checkpoint.

Arguments: resemble-ai/chatterbox checkout, directory containing the original
ve.safetensors/t3_cfg.safetensors/tokenizer.json, output JSON. Weights stay outside
the repository. The native checker consumes the same immutable checkpoint files.
"""
import gc
import json
from pathlib import Path
import subprocess
import sys
import types
import numpy as np
import torch
from safetensors.torch import load_file

checkout=Path(sys.argv[1]).resolve();weights=Path(sys.argv[2]).resolve();output_path=Path(sys.argv[3]).resolve()
for name,path in [('chatterbox',checkout/'src/chatterbox'),('chatterbox.models',checkout/'src/chatterbox/models')]:
    package=types.ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
from chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder
from chatterbox.models.t3.t3 import T3
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.tokenizers.tokenizer import EnTokenizer

torch.set_num_threads(1)
tokenizer=EnTokenizer(str(weights/'tokenizer.json'))
texts=['Hello, world!', '  spaced   words  ', "Don't stop...", 'Café déjà vu.', '[START]hello[STOP]',
    '0123456789 & $3.50', 'hello\nworld', 'こんにちは', 'English — punctuation!']
data=dict(source='Original ResembleAI/chatterbox ve + t3_cfg released checkpoints and author modules',
    revision='5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18',
    code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip(),
    tokenizer_cases=[dict(text=text,ids=tokenizer.encode(text)) for text in texts])
voice=VoiceEncoder().eval()
voice.load_state_dict(load_file(str(weights/'ve.safetensors')),strict=True)
voice_cases=[]
for length in [16000,40000]:
    time=np.arange(length,dtype=np.float32)
    audio=(.18*np.sin(time*.087)+.07*np.cos(time*.034)).astype(np.float32)
    audio[:length//6]=0;audio[-length//7:]=0
    embedding=voice.embeds_from_wavs([audio],sample_rate=16000)[0]
    voice_cases.append(dict(audio=audio.tolist(),embedding=embedding.tolist()))
data['voice_cases']=voice_cases
speaker=torch.tensor(voice_cases[-1]['embedding'],dtype=torch.float32)[None,:]
del voice;gc.collect()
print('Released voice encoder complete',flush=True)

model=T3().float().eval()
model.load_state_dict(load_file(str(weights/'t3_cfg.safetensors')),strict=True)
text_ids=[model.hp.start_text_token]+tokenizer.encode('Hello, world!')+[model.hp.stop_text_token]
speech_ids=[model.hp.start_speech_token,100,200,300,model.hp.stop_speech_token]
prompt_ids=[10,40,50,100]
text=torch.tensor([text_ids]);speech=torch.tensor([speech_ids]);prompt=torch.tensor([prompt_ids])
flat=lambda tensor:tensor.detach().flatten().tolist()
data.update(text=text_ids,speech=speech_ids,prompt=prompt_ids,speaker=flat(speaker))
for dtype,name in [(torch.float32,'float32'),(torch.float64,'float64')]:
    model.to(dtype=dtype)
    condition=T3Cond(speaker_emb=speaker.to(dtype),cond_prompt_speech_tokens=prompt,
        emotion_adv=torch.tensor([[[.7]]],dtype=dtype))
    with torch.inference_mode():
        result=model(t3_cond=condition,text_tokens=text,text_token_lens=torch.tensor([len(text_ids)]),
            speech_tokens=speech,speech_token_lens=torch.tensor([len(speech_ids)]))
    data[name]=dict(text_logits=flat(result.text_logits),speech_logits=flat(result.speech_logits),
        speech_argmax=result.speech_logits.argmax(-1).flatten().tolist())
    print('Released T3 complete:',name,flush=True)
output_path.parent.mkdir(parents=True,exist_ok=True)
output_path.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print('Wrote',output_path,flush=True)
