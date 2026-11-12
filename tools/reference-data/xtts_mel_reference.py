"""Execute Coqui's original wav_to_mel_cloning at both released frontend settings.

Uses torch float64 defaults so window/filterbank/FFT arithmetic has matching precision.
The original function is extracted unchanged; norm='slaney' retains torchaudio's
default HTK spacing, which differs from a Slaney-spaced librosa filterbank.
"""
import ast
import json
from pathlib import Path
import sys
import torch
import torchaudio

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(1434)
source=Path(sys.argv[1])/'TTS/tts/models/xtts.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
namespace=dict(torch=torch,torchaudio=torchaudio)
function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='wav_to_mel_cloning')
exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
norms=torch.linspace(.7,1.3,80)
samples=torch.arange(4096)
signals=[.04*torch.randn(4096)+.18*torch.sin(samples*.087)+.07*torch.cos(samples*.034),
    torch.randn(4353)*1e-8]
cases=[]
for fft,window in [(2048,1024),(1024,1024)]:
    for audio in signals:
        mel=namespace['wav_to_mel_cloning'](audio[None,:],mel_norms=norms,n_fft=fft,hop_length=256,
            win_length=window,power=2,normalized=False,sample_rate=22050,f_min=0,f_max=8000,n_mels=80)
        cases.append(dict(fft=fft,window=window,hop=256,audio=audio.tolist(),frames=mel.shape[2],mel=mel.flatten().tolist()))
destination=Path(__file__).resolve().parents[2]/'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/xtts_mel_reference.json'
destination.write_text(json.dumps(dict(source='Unmodified Coqui wav_to_mel_cloning; float64 torchaudio HTK spacing / Slaney area normalization',
    norms=norms.tolist(),cases=cases),separators=(',',':')),encoding='utf-8')
print('Wrote',len(cases),'original XTTS mel transforms')
