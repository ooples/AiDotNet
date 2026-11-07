"""Community VALL-E-X full checkpoint evidence; not Microsoft author weights.

Arguments: Plachtaa/VALL-E-X checkout, original checkpoint, output JSON.
Runs unchanged inference, controlling only sampling to expose two generated
frames and all seven NAR stages. Hooks observe original projection outputs.
Original float32, original float64 with float32-derived positional tables, and
float64 with recomputed nonlearned positional tables are recorded separately.
"""
import argparse, json, math, pathlib, sys
import torch

checkout, checkpoint, destination = map(pathlib.Path, sys.argv[1:4])
sys.path.insert(0, str(checkout.resolve()))
import models.vallex as original
from modules.embedding import SinePositionalEmbedding

torch.set_num_threads(1)
torch.serialization.add_safe_globals([pathlib.WindowsPath, argparse.Namespace])
package = torch.load(checkpoint, weights_only=True, mmap=True, map_location='cpu')
state = package['model']
model = original.VALLE(1024, 16, 12, norm_first=True, add_prenet=False,
    prefix_mode=1, share_embedding=True, nar_scale_factor=1, prepend_bos=True, num_quantizers=8).eval()
model.load_state_dict(state, strict=True)
flat = lambda x: x.detach().flatten().tolist()
text = torch.tensor([[9, 17, 22, 31, 7, 42, 13]])
prompt = ((torch.arange(24).reshape(1, 3, 8) * 31 + 11) % 1024)
results = dict(source='Community Plachtaa/VALL-E-X 3faaf8ccadb154d63b38070caf518ce9309ea0f4; not Microsoft author release',
    checkpoint_tensors=len(state), text=flat(text), prompt=flat(prompt), prompt_frames=3,
    text_languages=[0, 0, 1, 1, 1, 1, 1])

def execute():
    ar, nar = [], [[] for _ in range(7)]
    hooks = [model.ar_predict_layer.register_forward_hook(lambda m, a, out: ar.append(flat(out)))]
    for i, layer in enumerate(model.nar_predict_layers):
        hooks.append(layer.register_forward_hook(lambda m, a, out, i=i: nar[i].append(flat(out))))
    samples = iter([7, 11, 1024])
    previous = original.topk_sampling
    def controlled(logits, **kwargs):
        token = next(samples)
        return torch.full((1, 1), token, dtype=torch.long), torch.zeros(1, dtype=logits.dtype)
    original.topk_sampling = controlled
    try:
        with torch.no_grad():
            codes = model.inference(text, torch.tensor([7]), prompt, torch.tensor(2),
                prompt_language='en', text_language='zh')
    finally:
        original.topk_sampling = previous
        for hook in hooks: hook.remove()
    assert len(ar) == 3 and all(len(x) == 1 for x in nar)
    return dict(ar_logits=ar, nar_logits=[x[0] for x in nar], codes=flat(codes), shape=list(codes.shape))

results['original_float32'] = execute()
model.double()
results['original_float64_float32_positions'] = execute()
for module in model.modules():
    if isinstance(module, SinePositionalEmbedding):
        # Only regenerate nonlearned PE. All learned weights and neural methods
        # remain the original source, matching native double sinusoid arithmetic.
        position = torch.arange(4000, dtype=torch.float64)[:, None]
        frequency = torch.exp(torch.arange(0, 1024, 2, dtype=torch.float64) * -(math.log(10000.0)/1024))
        pe = torch.empty(4000, 1024, dtype=torch.float64)
        pe[:, 0::2] = torch.sin(position * frequency)
        pe[:, 1::2] = torch.cos(position * frequency)
        module.pe = pe[None]
results['float64_derived_positions'] = execute()
destination.write_text(json.dumps(results), encoding='utf-8')
print('Generated full community VALL-E-X AR and seven NAR-stage logits with unchanged inference.', flush=True)
