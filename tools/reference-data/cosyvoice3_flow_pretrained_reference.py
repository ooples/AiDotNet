"""Original CosyVoice3 pretrained causal DiT flow, with controlled original noise.

Arguments: author checkout, pinned release directory, output JSON. All forward and
Euler methods execute unchanged. Only optional serving imports and BASECFM's unused
construction are excluded; the actual full DiT and wrapper state load strictly.
"""
import ast
import json
import logging
from pathlib import Path
import random
import subprocess
import sys
from typing import Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig

checkout=Path(sys.argv[1]).resolve();weights=Path(sys.argv[2]).resolve();destination=Path(sys.argv[3]).resolve()
sys.path.insert(0,str(checkout));torch.set_num_threads(1)
from cosyvoice.flow.DiT.dit import DiT
from cosyvoice.utils.mask import make_pad_mask

def original_class(path,name):
    return next(node for node in ast.parse(path.read_text(encoding='utf-8')).body
        if isinstance(node,ast.ClassDef) and node.name==name)

lookahead_path=checkout/'cosyvoice/transformer/upsample_encoder.py'
exec(compile(ast.fix_missing_locations(ast.Module(body=[original_class(lookahead_path,'PreLookaheadLayer')],type_ignores=[])),str(lookahead_path),'exec'),globals())
cfm_path=checkout/'cosyvoice/flow/flow_matching.py'
conditional=original_class(cfm_path,'ConditionalCFM')
conditional.bases=[ast.Attribute(value=ast.Attribute(value=ast.Name(id='torch',ctx=ast.Load()),attr='nn',ctx=ast.Load()),attr='Module',ctx=ast.Load())]
conditional.body=[node for node in conditional.body if isinstance(node,ast.FunctionDef)
    and node.name in ['solve_euler','forward_estimator','compute_loss']]
causal=original_class(cfm_path,'CausalConditionalCFM')
causal.body=[node for node in causal.body if isinstance(node,ast.FunctionDef) and node.name=='forward']
exec(compile(ast.fix_missing_locations(ast.Module(body=[conditional,causal],type_ignores=[])),str(cfm_path),'exec'),globals())
flow_path=checkout/'cosyvoice/flow/flow.py';online_feature=False
exec(compile(ast.fix_missing_locations(ast.Module(body=[original_class(flow_path,'CausalMaskedDiffWithDiT')],type_ignores=[])),str(flow_path),'exec'),globals())
solver=CausalConditionalCFM();solver.inference_cfg_rate=.7;solver.training_cfg_rate=.2;solver.sigma_min=1e-6;solver.t_scheduler='cosine'
solver.estimator=DiT(dim=1024,depth=22,heads=16,dim_head=64,ff_mult=2,mel_dim=80,mu_dim=80,spk_dim=80,
    out_channels=80,static_chunk_size=50,num_decoding_left_chunks=-1)
# This is the author's exact seeded rand_noise construction, not a replacement sampler.
torch.manual_seed(0);solver.rand_noise=torch.randn([1,80,50*300])
model=CausalMaskedDiffWithDiT(input_size=80,output_size=80,spk_embed_dim=192,output_type='mel',vocab_size=6561,
    input_frame_rate=25,only_mask_loss=True,token_mel_ratio=2,pre_lookahead_len=3,
    pre_lookahead_layer=PreLookaheadLayer(in_channels=80,channels=1024,pre_lookahead_len=3),decoder=solver).eval()
model.load_state_dict(torch.load(weights/'flow.pt',map_location='cpu',weights_only=True),strict=True)
torch.manual_seed(3304)
tokens=torch.tensor([[(i*137+8)%6561 for i in range(26)]]);prompt=torch.tensor([[2,19]])
speaker=torch.randn(1,192);prompt_mel=torch.randn(1,4,80)*.2
flat=lambda value:value.detach().flatten().tolist()
data=dict(revision='29e01c4e8d000f4bcd70751be16fa94bf3d85a18',
    code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip(),
    tokens=flat(tokens),prompt_tokens=flat(prompt),speaker=flat(speaker),prompt_mel=flat(prompt_mel),
    noise=flat(solver.rand_noise[:,:,:56]))
for dtype,label in [(torch.float32,'float32'),(torch.float64,'float64'),(torch.float64,'float64_rotary_buffer')]:
    model.to(dtype)
    if label=='float64_rotary_buffer':
        # Independent precision diagnostic: only this derived, non-learned buffer changes precision.
        # Original float32/upcast-float64 results above remain intact; every neural method is unchanged.
        model.decoder.estimator.rotary_embed.inv_freq=1.0/(10000.0**(torch.arange(0,64,2,dtype=torch.float64)/64))
    with torch.inference_mode():
        encoded=model.pre_lookahead_layer(model.input_embedding(torch.cat([prompt,tokens],dim=1))).repeat_interleave(2,dim=1)
        projected=model.spk_embed_affine_layer(F.normalize(speaker.to(dtype),dim=1))
        results={}
        for streaming in [False,True]:
            captured=[]
            unrounded=[]
            def trace_solver_return(frame,event,arg):
                if event=='return':unrounded.append(frame.f_locals['sol'][-1].detach().clone())
                return trace_solver_return
            def trace_solver_call(frame,event,arg):
                if frame.f_code.co_name=='solve_euler' and frame.f_code.co_filename==str(cfm_path):return trace_solver_return
                return None
            hook=solver.register_forward_hook(lambda module,args,result:captured.append(result[0].clone()))
            previous_trace=sys.gettrace()
            try:
                # Read the final solver local before its mandatory .float() return cast. No method is replaced.
                sys.settrace(trace_solver_call)
                output,_=model.inference(tokens,torch.tensor([26]),prompt,torch.tensor([2]),prompt_mel.to(dtype),
                    torch.tensor([4]),speaker.to(dtype),streaming=streaming,finalize=True)
            finally:sys.settrace(previous_trace)
            hook.remove()
            if len(unrounded)!=1:raise RuntimeError('Original solver return trace was not captured exactly once')
            results[str(streaming).lower()]=dict(output=flat(output),untrimmed=flat(unrounded[0]),solver_output=flat(captured[0]))
    data[label]=dict(encoded=flat(encoded),projected_speaker=flat(projected),outputs=results)
    print('Original complete pretrained CosyVoice3 flow complete:',label,flush=True)
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print('Wrote original pretrained CosyVoice3 flow reference:',destination,flush=True)
