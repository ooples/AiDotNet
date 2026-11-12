"""Independent PyTorch FFT-block reference for the FastSpeech-family paper equations.

Uses PyTorch MultiheadAttention/Conv1d/LayerNorm and autograd, with QKV matrices,
output affine projection, post-normalization, ReLU and the published kernel pairs.
This is an equation reference, not an unreleased author checkpoint.
"""
import json
from pathlib import Path
import torch

class FftBlock(torch.nn.Module):
    def __init__(self,kernels):
        super().__init__()
        self.attention=torch.nn.MultiheadAttention(8,2,bias=False,batch_first=True)
        self.attention.out_proj=torch.nn.Linear(8,8)
        self.norm1=torch.nn.LayerNorm(8)
        self.conv1=torch.nn.Conv1d(8,16,kernels[0],padding=kernels[0]//2)
        self.conv2=torch.nn.Conv1d(16,8,kernels[1],padding=kernels[1]//2)
        self.norm2=torch.nn.LayerNorm(8)
    def forward(self,value):
        h=self.norm1(value+self.attention(value,value,value,need_weights=False)[0])
        return self.norm2(h+self.conv2(self.conv1(h.transpose(1,2)).relu()).transpose(1,2))

torch.manual_seed(1905)
torch.set_num_threads(1)
flat=lambda value:value.detach().flatten().tolist()
cases=[]
for kernels in [(3,3),(9,1)]:
    block=FftBlock(kernels).double().eval()
    value=torch.randn(2,5,8,dtype=torch.float64,requires_grad=True)
    target=torch.randn_like(value)
    output=block(value)
    loss=(output-target).square().mean()
    loss.backward()
    cases.append(dict(kernels=kernels,weights={key:flat(parameter) for key,parameter in block.state_dict().items()},
        input=flat(value),target=flat(target),output=flat(output),loss=loss.item(),input_gradient=flat(value.grad),
        gradients={key:flat(parameter.grad) for key,parameter in block.named_parameters()}))
root=Path(__file__).resolve().parents[2]/'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
scale=torch.nn.Linear(6,8,bias=False).double()
shift=torch.nn.Linear(6,8,bias=False).double()
value=torch.randn(2,5,8,dtype=torch.float64,requires_grad=True)
condition=torch.randn(2,6,dtype=torch.float64,requires_grad=True)
target=torch.randn_like(value)
output=torch.nn.functional.layer_norm(value,(8,),eps=1e-5)*scale(condition)[:,None,:]+shift(condition)[:,None,:]
loss=(output-target).square().mean()
loss.backward()
conditional=dict(input=flat(value),condition=flat(condition),target=flat(target),scale_weight=flat(scale.weight),
    shift_weight=flat(shift.weight),output=flat(output),loss=loss.item(),input_gradient=flat(value.grad),
    condition_gradient=flat(condition.grad),scale_gradient=flat(scale.weight.grad),shift_gradient=flat(shift.weight.grad))
(root/'classic_fft_reference.json').write_text(json.dumps(dict(
    source='FastSpeech section 3.1 / FastSpeech2 Appendix A / AdaSpeech Eq. 1; independent PyTorch operators',cases=cases,
    conditional=conditional),
    separators=(',',':')),encoding='utf-8')
print('Two complete FFT block outputs and all gradients')
