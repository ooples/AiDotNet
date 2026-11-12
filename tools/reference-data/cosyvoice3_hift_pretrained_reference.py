"""Original pretrained CausalHiFT inference, preserving actual stochastic excitation.

Arguments: author checkout, pinned weight directory, author flow reference, output JSON.
The native decoder receives the identical source tensor, separating waveform-network
arithmetic from different random-number generators. No author methods are replaced.
"""
import json
from pathlib import Path
import subprocess
import sys
import torch

checkout=Path(sys.argv[1]).resolve();weights=Path(sys.argv[2]).resolve();flow_path=Path(sys.argv[3]);destination=Path(sys.argv[4])
sys.path.insert(0,str(checkout));torch.set_num_threads(1)
from cosyvoice.hifigan.generator import CausalHiFTGenerator
from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
model=CausalHiFTGenerator(in_channels=80,base_channels=512,nb_harmonics=8,sampling_rate=24000,
    nsf_alpha=.1,nsf_sigma=.003,nsf_voiced_threshold=10,upsample_rates=[8,5,3],upsample_kernel_sizes=[16,11,7],
    istft_params=dict(n_fft=16,hop_len=4),resblock_kernel_sizes=[3,7,11],resblock_dilation_sizes=[[1,3,5]]*3,
    source_resblock_kernel_sizes=[7,7,11],source_resblock_dilation_sizes=[[1,3,5]]*3,lrelu_slope=.1,audio_limit=.99,
    conv_pre_look_right=4,f0_predictor=CausalConvRNNF0Predictor(num_class=1,in_channels=80,cond_channels=512)).eval()
model.load_state_dict(torch.load(weights/'hift.pt',map_location='cpu',weights_only=True),strict=True)
flow=json.loads(flow_path.read_text(encoding='utf-8'))
mel=torch.tensor(flow['float64']['outputs']['false']['output'],dtype=torch.float64).reshape(1,80,52)
flat=lambda x:x.detach().flatten().tolist()
data=dict(revision=flow['revision'],code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip(),
    mel=flat(mel),frames=52)
for dtype,label in [(torch.float32,'float32'),(torch.float64,'float64')]:
    model.to(dtype);torch.manual_seed(3305)
    captured=[]
    hook=model.f0_predictor.register_forward_hook(lambda module,args,result:captured.append(result.clone()))
    with torch.inference_mode():waveform,source=model.inference(mel.to(dtype),finalize=True)
    hook.remove()
    data[label]=dict(f0=flat(captured[0]),source=flat(source),waveform=flat(waveform))
    print('Original pretrained CosyVoice3 CausalHiFT complete:',label,flush=True)
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print('Wrote original pretrained CausalHiFT reference:',destination,flush=True)
