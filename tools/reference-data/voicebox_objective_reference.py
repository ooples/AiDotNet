"""Independent PyTorch implementation of Voicebox's OT-CFM and Appendix A duration equations.

This is an equation reference, not execution of unavailable Meta Voicebox source code.
"""
import json
from pathlib import Path
import torch
torch.manual_seed(762)
flat=lambda x:x.detach().flatten().tolist()
target=torch.randn(5,3,dtype=torch.float64,requires_grad=True)
prior=torch.randn(5,3,dtype=torch.float64,requires_grad=True)
prediction=torch.randn(5,3,dtype=torch.float64,requires_grad=True)
time=.37; sigma=1e-5
state=(1-(1-sigma)*time)*prior+time*target
velocity=target-(1-sigma)*prior
mask=torch.tensor([True,False,True,False,True])
loss=(prediction[mask]-velocity[mask]).square().mean()
loss.backward()
data=dict(source='Independent Voicebox OT-CFM equations and Appendix A duration transform',
    target=flat(target),prior=flat(prior),prediction=flat(prediction),state=flat(state),velocity=flat(velocity),
    loss=loss.item(), target_gradient=flat(target.grad),prior_gradient=flat(prior.grad),prediction_gradient=flat(prediction.grad),
    duration_cases=[dict(frames=d,uniform=u,value=torch.log1p(torch.tensor(d+u-.5,dtype=torch.float64)).item())
        for d in [0,1,10] for u in [0.,.25,.5,.999]])
out=Path(__file__).resolve().parents[2]/'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/voicebox_objective_reference.json'
out.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
