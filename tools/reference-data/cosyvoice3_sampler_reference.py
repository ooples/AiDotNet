"""Execute the unmodified CosyVoice CFM Euler solver around its real DiT estimator.

Argument: FunAudioLLM/CosyVoice checkout. Only unrelated BASECFM construction and
optional inference backends are excluded when loading the original class methods.
"""
import ast
import json
from pathlib import Path
import sys
import torch

checkout = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(checkout))
from cosyvoice.flow.DiT.dit import DiT
tree = ast.parse((checkout / 'cosyvoice/flow/flow_matching.py').read_text(encoding='utf-8'))
classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
conditional = next(node for node in classes if node.name == 'ConditionalCFM')
conditional.bases = [ast.Attribute(value=ast.Name(id='torch', ctx=ast.Load()), attr='nn', ctx=ast.Load())]
conditional.bases = [ast.Attribute(value=conditional.bases[0], attr='Module', ctx=ast.Load())]
conditional.body = [node for node in conditional.body if isinstance(node, ast.FunctionDef)
    and node.name in ['solve_euler', 'forward_estimator', 'compute_loss']]
causal = next(node for node in classes if node.name == 'CausalConditionalCFM')
causal.body = [node for node in causal.body if isinstance(node, ast.FunctionDef) and node.name == 'forward']
environment = dict(torch=torch, F=torch.nn.functional)
exec(compile(ast.fix_missing_locations(ast.Module(body=[conditional, causal], type_ignores=[])),
    str(checkout / 'cosyvoice/flow/flow_matching.py'), 'exec'), environment)
torch.manual_seed(3303)
torch.set_num_threads(1)
model = DiT(dim=32, depth=1, heads=2, dim_head=16, ff_mult=2, mel_dim=80, mu_dim=80,
    spk_dim=80, out_channels=80, dropout=0, static_chunk_size=2, num_decoding_left_chunks=-1).double().eval()
with torch.no_grad():
    for parameter in model.parameters(): parameter.normal_(0, .1)
solver = environment['CausalConditionalCFM']()
solver.estimator = model
solver.inference_cfg_rate = .7
solver.training_cfg_rate = 0
solver.sigma_min = 1e-6
mu = torch.randn(1, 80, 6, dtype=torch.float64)
speaker = torch.randn(1, 80, dtype=torch.float64)
condition = torch.randn(1, 80, 6, dtype=torch.float64)
noise = torch.randn(1, 80, 6, dtype=torch.float64)
time = 1 - torch.cos(torch.linspace(0, 1, 5, dtype=torch.float64) * .5 * torch.pi)
flat = lambda tensor: tensor.detach().flatten().tolist()
with torch.no_grad():
    outputs = {str(streaming).lower(): flat(solver.solve_euler(noise.clone(), time,
        mu, torch.ones(1, 1, 6), speaker, condition, streaming=streaming).transpose(1, 2))
        for streaming in [False, True]}
root = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
(root / 'cosyvoice3_sampler_reference.json').write_text(json.dumps(dict(
    source='FunAudioLLM/CosyVoice ConditionalCFM.solve_euler + DiT, original methods',
    weights={key: flat(value) for key, value in model.state_dict().items()},
    mu=flat(mu.transpose(1, 2)), speaker=flat(speaker), condition=flat(condition.transpose(1, 2)),
    noise=flat(noise.transpose(1, 2)), outputs=outputs), separators=(',', ':')), encoding='utf-8')
print('Wrote real CosyVoice3 four-step guided Euler trajectories')
