"""Execute the original Chatterbox waveform-to-voice-embedding pipeline.

Argument: resemble-ai/chatterbox checkout. LSTM widths are reduced, but the
unmodified mel, silence trimming, partial windows and three-layer encoder execute.
"""
import json
from pathlib import Path
import sys
import types
import numpy as np
import torch

checkout = Path(sys.argv[1]).resolve()
for name,path in [('chatterbox',checkout/'src/chatterbox'),('chatterbox.models',checkout/'src/chatterbox/models')]:
    package=types.ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
from chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder
from chatterbox.models.voice_encoder.config import VoiceEncConfig
from chatterbox.models.voice_encoder.melspec import melspectrogram
import librosa
torch.manual_seed(3256)
torch.set_num_threads(1)
hp=VoiceEncConfig();hp.num_mels=4;hp.ve_hidden_size=8;hp.speaker_embed_size=6
model=VoiceEncoder(hp).eval()
flat=lambda value:value.detach().flatten().tolist()
cases=[]
for frames,rate in [(160,1.3),(8,500.0)]:
    hp.ve_partial_frames=frames
    for length in [1,201,1280,16000,40000,-4000]:
        quiet = length < 0
        length = abs(length)
        index=np.arange(length,dtype=np.float32)
        wav=(.18*np.sin(index*.087)+.07*np.cos(index*.034)).astype(np.float32)
        if length>1280:
            wav[:length//6]=0;wav[-length//7:]=0
        if quiet: wav *= 1e-6
        embedding=model.embeds_from_wavs([wav],sample_rate=16000,rate=rate)[0]
        trimmed=librosa.effects.trim(wav,top_db=20)[0]
        mel=melspectrogram(trimmed,hp).T
        cases.append(dict(partial_frames=frames,partial_rate=rate,audio=wav.tolist(),embedding=embedding.tolist(),
            trimmed_length=len(trimmed),mel=mel.flatten().tolist(),mel_frames=len(mel)))
root=Path(__file__).resolve().parents[2]/'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
(root/'chatterbox_voice_reference.json').write_text(json.dumps(dict(
    source='resemble-ai/chatterbox VoiceEncoder.embeds_from_wavs, original librosa mel/trim/partials/LSTM',
    weights={key:flat(value) for key,value in model.state_dict().items()},
    weight_shapes={key:list(value.shape) for key,value in model.state_dict().items()},cases=cases),separators=(',',':')),encoding='utf-8')
print(len(cases),'complete original voice-encoder cases')
