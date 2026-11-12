"""Original VoiceCraft GigaSpeech AudioCraft codec, including trained EMA VQ buffers.

Arguments: AudioCraft 1.3.0 source root, pinned VoiceCraft weight directory, output JSON.
Neural classes and the original compression-model factory execute unchanged. Optional
music-model/package initialization is excluded; distributed-training callbacks raise
if reached, so no inference result can be supplied by a substitute implementation.
"""
import ast
import collections
import gc
import json
from pathlib import Path
import sys
import types
import typing
import torch
import omegaconf
from omegaconf import DictConfig,ListConfig,OmegaConf
from omegaconf.base import ContainerMetadata,Metadata
from omegaconf.nodes import AnyNode

checkout=Path(sys.argv[1]).resolve();weights=Path(sys.argv[2]).resolve();destination=Path(sys.argv[3]).resolve()
torch.set_num_threads(1)
for name,relative in [('audiocraft','audiocraft'),('audiocraft.models','audiocraft/models'),('audiocraft.modules','audiocraft/modules')]:
    module=types.ModuleType(name);module.__path__=[str(checkout/relative)];sys.modules[name]=module
    if '.' in name:setattr(sys.modules[name.rsplit('.',1)[0]],name.rsplit('.',1)[1],module)
def unavailable(*args,**kwargs):raise RuntimeError('A distributed-training callback was reached during codec inference')
flashy=types.ModuleType('flashy');flashy.distrib=types.SimpleNamespace(broadcast_tensors=unavailable,all_reduce=unavailable,
    world_size=unavailable,rank=unavailable,is_distributed=unavailable)
sys.modules['flashy']=flashy
import audiocraft
from audiocraft.modules.seanet import SEANetEncoder,SEANetDecoder
audiocraft.modules.SEANetEncoder=SEANetEncoder;audiocraft.modules.SEANetDecoder=SEANetDecoder
from audiocraft.models.encodec import CompressionModel,EncodecModel
import audiocraft.quantization as qt

def original_function(path,name):
    return next(node for node in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(node,ast.FunctionDef) and node.name==name)
utilities=checkout/'audiocraft/utils/utils.py'
exec(compile(ast.fix_missing_locations(ast.Module(body=[original_function(utilities,'dict_from_config')],type_ignores=[])),str(utilities),'exec'),globals())
factory=checkout/'audiocraft/models/builders.py'
exec(compile(ast.fix_missing_locations(ast.Module(body=[original_function(factory,name) for name in
    ['get_quantizer','get_encodec_autoencoder','get_compression_model']],type_ignores=[])),str(factory),'exec'),globals())
metadata=[DictConfig,ListConfig,ContainerMetadata,Metadata,AnyNode,collections.defaultdict,typing.Any,int,dict,list]
with torch.serialization.safe_globals(metadata):
    package=torch.load(weights/'encodec_4cb2048_giga.th',map_location='cpu',weights_only=True,mmap=True)
cfg=package['xp.cfg'];cfg.device='cpu'
model=get_compression_model(cfg).eval()
model.load_state_dict(package['best_state']['model'],strict=True)
data=dict(revision='52b0f909d652ea016966804767320f13ede39abd',source='Original AudioCraft 1.3.0 factories / EncodecModel / SEANet / RVQ; best_state.model',
    config={name:OmegaConf.to_container(getattr(cfg,name),resolve=True) for name in ['encodec','seanet','rvq']},cases=[])
del package;gc.collect()
torch.manual_seed(3308)
for samples in [1,3201,16000]:
    time=torch.arange(samples,dtype=torch.float32)
    audio=.04*torch.randn(samples)+.18*torch.sin(time*.087)+.07*torch.cos(time*.034)
    data['cases'].append(dict(audio=audio.tolist()))
flat=lambda value:value.detach().flatten().tolist()
for dtype,label in [(torch.float32,'float32'),(torch.float64,'float64')]:
    model.to(dtype)
    for item in data['cases']:
        audio=torch.tensor(item['audio'],dtype=dtype).reshape(1,1,-1)
        with torch.inference_mode():
            preprocessed,_=model.preprocess(audio)
            encoded=model.encoder(preprocessed)
            codes,scale=model.encode(audio)
            latent=model.decode_latent(codes)
            waveform=model.decode(codes,scale)
        item[label]=dict(encoded=flat(encoded),codes=flat(codes),code_shape=list(codes.shape),latent=flat(latent),
            waveform=flat(waveform),scale=None if scale is None else flat(scale))
    print('Original complete pretrained GigaSpeech EnCodec complete:',label,flush=True)
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print('Wrote original pretrained VoiceCraft codec reference:',destination,flush=True)
