"""Original Coqui HifiDecoder forward and wrapped named-checkpoint import regression."""
import json
from pathlib import Path
import sys
import types
import torch

checkout=Path(sys.argv[1]).resolve();package=checkout/'TTS'
for name,path in [('TTS',package),('TTS.tts',package/'tts'),('TTS.tts.layers',package/'tts/layers'),('TTS.tts.layers.xtts',package/'tts/layers/xtts')]:
    module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
module=types.ModuleType('TTS.utils.io')
def unavailable(*args,**kwargs): raise RuntimeError('No implicit checkpoint IO in reference forward')
module.load_fsspec=unavailable;sys.modules[module.__name__]=module
from TTS.tts.layers.xtts.hifigan_decoder import HifiDecoder
torch.set_num_threads(1);torch.manual_seed(374)
model=HifiDecoder(decoder_input_dim=4,resblock_dilation_sizes_decoder=[[1,3,5]],resblock_kernel_sizes_decoder=[3],
    upsample_rates_decoder=[2,2],upsample_initial_channel_decoder=16,upsample_kernel_sizes_decoder=[4,4],d_vector_dim=4).double().eval()
latents=torch.randn(1,3,4,dtype=torch.float64);speaker=torch.randn(1,4,1,dtype=torch.float64)
with torch.inference_mode(): waveform=model(latents,g=speaker)
state={key.replace('.parametrizations.weight.original0','.weight_g').replace('.parametrizations.weight.original1','.weight_v'):value
    for key,value in model.waveform_decoder.state_dict().items()}
destination=Path(__file__).resolve().parents[2]/'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
torch.save(dict(model={'hifigan_decoder.waveform_decoder.'+key:value for key,value in state.items()},step=1),
    destination/'xtts_wrapped_decoder_reference.pth')
flat=lambda tensor:tensor.detach().flatten().tolist()
(destination/'xtts_decoder_reference.json').write_text(json.dumps(dict(source='Unmodified Coqui HifiDecoder.forward, global cond_layer and per-stage conds; original legacy weight norm names',
    latents=flat(latents),speaker=flat(speaker),waveform=flat(waveform)),separators=(',',':')),encoding='utf-8')
print('Wrote original Coqui decoder waveform and',len(state),'wrapped named tensors')
