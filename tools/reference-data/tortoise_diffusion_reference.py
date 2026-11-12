"""Run the original TorToiSe DiffusionTts model. Argument: neonbjb/tortoise-tts checkout."""
import json
from pathlib import Path
import sys
import types
import torch

root = Path(sys.argv[1]).resolve()
package = types.ModuleType('tortoise')
package.__path__ = [str(root / 'tortoise')]
sys.modules['tortoise'] = package
from tortoise.models.diffusion_decoder import DiffusionTts
from tortoise.utils.diffusion import GaussianDiffusion, SpacedDiffusion, space_timesteps, get_named_beta_schedule

torch.manual_seed(468)
model = DiffusionTts(model_channels=32, num_layers=1, in_channels=4,
    in_latent_channels=8, in_tokens=9, out_channels=8, dropout=0,
    num_heads=4, layer_drop=0, unconditioned_percentage=0).eval()
with torch.no_grad():
    for name, parameter in model.named_parameters():
        if name.endswith('proj_out.weight'):
            parameter.normal_(0, .02)
    noisy = torch.randn(1, 4, 9)
    latents = torch.randn(1, 3, 8)
    mel = torch.randn(1, 4, 12)
    condition = model.get_conditioning(mel)
    aligned = model.timestep_independent(latents, condition, 9, False)
    timestep = torch.tensor([137])
    prediction = model(noisy, timestep, latents, condition)
    unconditioned = model(noisy, timestep, latents, condition, conditioning_free=True)
    codes = torch.tensor([[1, 3, 4]])
    code_prediction = model(noisy, timestep, codes, condition)
flat = lambda t: t.detach().flatten().tolist()
data = dict(source='neonbjb/tortoise-tts/tortoise/models/diffusion_decoder.py DiffusionTts',
    weights={k: flat(v) for k, v in model.state_dict().items()},
    weight_shapes={k: list(v.shape) for k, v in model.state_dict().items()},
    noisy=flat(noisy), latents=flat(latents), mel=flat(mel), condition=flat(condition),
    aligned=flat(aligned), prediction=flat(prediction), unconditioned=flat(unconditioned),
    codes=[1, 3, 4], code_prediction=flat(code_prediction), timestep=137)
process = GaussianDiffusion(betas=get_named_beta_schedule('linear', 4000), model_mean_type='epsilon',
    model_var_type='learned_range', loss_type='mse', conditioning_free=True, conditioning_free_k=2)
statistics = process.p_mean_variance(lambda x, t, conditioning_free=False:
    unconditioned if conditioning_free else prediction, noisy, timestep)
data['statistics'] = {k: flat(statistics[k]) for k in ('mean', 'log_variance', 'pred_xstart')}
data['posterior_variance'] = process.posterior_variance[137].item()
selected = sorted(space_timesteps(4000, [3]))
spaced = SpacedDiffusion(use_timesteps=set(selected), betas=get_named_beta_schedule('linear', 4000),
    model_mean_type='epsilon', model_var_type='learned_range', loss_type='mse',
    conditioning_free=True, conditioning_free_k=2)
data['spaced'] = []
for index in range(3):
    stats = spaced.p_mean_variance(lambda x, t, conditioning_free=False:
        unconditioned if conditioning_free else prediction, noisy, torch.tensor([index]))
    data['spaced'].append(dict(index=index, step=selected[index], previous=-1 if index == 0 else selected[index-1],
        clipped_variance=spaced.posterior_variance[max(1, index)].item(),
        mean=flat(stats['mean']), log_variance=flat(stats['log_variance']), clean=flat(stats['pred_xstart'])))
data['training_clean'] = flat((noisy * .2).clamp(-1, 1))
data['training_clean'][0] = -1
data['training_clean'][1] = 1
data['training_noise'] = flat(noisy * .3)
process.conditioning_free = False
data['training'] = []
for step in (0, 137):
    output = prediction.detach().clone().requires_grad_()
    clean = torch.tensor(data['training_clean']).reshape(1, 4, 9)
    noise = torch.tensor(data['training_noise']).reshape(1, 4, 9)
    terms = process.training_losses(lambda x, t: output, clean, torch.tensor([step]), noise=noise)
    terms['loss'].mean().backward()
    data['training'].append(dict(step=step, mse=terms['mse'].item(), vb=terms['vb'].item(),
        loss=terms['loss'].item(), gradient=flat(output.grad)))
model.zero_grad()
train_prediction = model(noisy, timestep, latents, model.get_conditioning(mel))
train_loss = train_prediction.square().mean()
train_loss.backward()
data['sgd_loss'] = train_loss.item()
data['sgd_rate'] = .01
with torch.no_grad():
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.sub_(data['sgd_rate'] * parameter.grad)
    data['sgd_prediction_after'] = flat(model(noisy, timestep, latents, model.get_conditioning(mel)))
out = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/tortoise_diffusion_reference.json'
out.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
print(out)
