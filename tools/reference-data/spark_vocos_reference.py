"""Run the original conditional and unconditional Spark VocosBackbone, including an SGD step.

Argument: Spark-TTS checkout. Also tests the actual Encoder's no-resampling
SamplingBlock residual sums, which are not the identity despite ratio=1.
"""
import json
from pathlib import Path
import sys
import types
import torch
root = Path(sys.argv[1]).resolve()
package = types.ModuleType('sparktts'); package.__path__ = [str(root / 'sparktts')]
sys.modules['sparktts'] = package
from sparktts.modules.blocks.vocos import VocosBackbone
from sparktts.modules.encoder_decoder.feat_encoder import Encoder
from sparktts.modules.encoder_decoder.feat_decoder import Decoder
from sparktts.modules.encoder_decoder.wave_generator import WaveGenerator
torch.set_num_threads(1)
flat = lambda t: t.detach().flatten().tolist()
out = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
cases=[]
for conditional in [False, True]:
    torch.manual_seed(510 + conditional)
    model = VocosBackbone(input_channels=4, dim=8, intermediate_dim=16, num_layers=2,
        condition_dim=6 if conditional else None).double().eval()
    # Make all conditioning projections nontrivial, independent of their initialization.
    with torch.no_grad():
        for k,p in model.named_parameters():
            if '.scale.' in k or '.shift.' in k: p.uniform_(-.3,.3)
    inputs=torch.randn(1,4,5,dtype=torch.float64)
    condition=torch.randn(1,6,dtype=torch.float64) if conditional else None
    run=lambda:model(inputs,condition).transpose(1,2)
    result=run(); loss=result.square().mean()
    case=dict(weights={k:flat(v) for k,v in model.state_dict().items()},
        weight_shapes={k:list(v.shape) for k,v in model.state_dict().items()}, input=flat(inputs),
        condition=flat(condition) if conditional else None, output=flat(result), loss=loss.item(), learning_rate=.01)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:p.sub_(.01*p.grad)
        case['after_sgd']=flat(run())
    cases.append(case)
(out/'spark_vocos_reference.json').write_text(json.dumps(dict(source='SparkAudio/Spark-TTS VocosBackbone',cases=cases),separators=(',',':')),encoding='utf-8')

torch.manual_seed(618)
encoder=Encoder(4,8,16,1,8).double().eval()
prenet=Decoder(8,8,16,1,8,condition_dim=6).double().eval()
postnet=Decoder(8,8,16,1,4).double().eval()
wave=WaveGenerator(8,32,[2,2],[4,4]).double().eval()
inputs=torch.randn(1,4,5,dtype=torch.float64)
condition=torch.randn(1,6,dtype=torch.float64)
encoded=encoder(inputs)
pre=prenet(encoded,condition)
post=postnet(pre)
wave_input=pre+condition.mean(-1,keepdim=True).unsqueeze(-1)
state={}
for prefix,model in [('encoder.',encoder),('prenet.',prenet),('postnet.',postnet),('decoder.',wave)]:
    state.update({prefix+k:v for k,v in model.state_dict().items()})
data=dict(source='SparkAudio/Spark-TTS Encoder, Decoder and WaveGenerator',
    weights={k:flat(v) for k,v in state.items()},weight_shapes={k:list(v.shape) for k,v in state.items()},
    input=flat(inputs),condition=flat(condition),encoded=flat(encoded),prenet=flat(pre),postnet=flat(post),
    wave_input=flat(wave_input),waveform=flat(wave(wave_input)))
(out/'spark_codec_modules_reference.json').write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
