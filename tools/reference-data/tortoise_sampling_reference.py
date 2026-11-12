"""Execute original TorToiSe's complete spaced DDPM, recording random draws only.

Arguments: author checkout; optionally released checkpoint directory and output JSON.
The complete author DiffusionTts forward and SpacedDiffusion sampler are unchanged.
Without optional arguments, weights are the existing author-generated reduced
diffusion fixture. The released mode imports the entire original trained checkpoint
and runs the standard 200-step schedule, recording its SHA-256 and source revision.
Both modes test sampling integration rather than perceptual quality.
"""
import json
from pathlib import Path
import subprocess
import sys
import types
import torch

checkout = Path(sys.argv[1]).resolve()
package = types.ModuleType('tortoise')
package.__path__ = [str(checkout / 'tortoise')]
sys.modules['tortoise'] = package
from tortoise.models.diffusion_decoder import DiffusionTts
from tortoise.utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule

torch.set_num_threads(1)
directory = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
released = len(sys.argv) == 4
torch.manual_seed(2294)
if released:
    model = DiffusionTts(model_channels=1024, num_layers=10, in_channels=100,
        in_latent_channels=1024, in_tokens=8193, out_channels=200, dropout=0,
        num_heads=16, layer_drop=0, unconditioned_percentage=0).eval()
    model.load_state_dict(torch.load(Path(sys.argv[2]) / '.models/diffusion_decoder.pth',
        map_location='cpu', weights_only=True), strict=True)
    mel = torch.randn(1, 100, 12) * .3
    latents = torch.randn(1, 3, 1024)
else:
    fixture = json.loads((directory / 'tortoise_diffusion_reference.json').read_text())
    model = DiffusionTts(model_channels=32, num_layers=1, in_channels=4,
        in_latent_channels=8, in_tokens=9, out_channels=8, dropout=0,
        num_heads=4, layer_drop=0, unconditioned_percentage=0).eval()
    model.load_state_dict({name: torch.tensor(values).reshape(fixture['weight_shapes'][name])
                          for name, values in fixture['weights'].items()}, strict=True)
    mel = torch.tensor(fixture['mel']).reshape(1, 4, 12)
    latents = torch.tensor(fixture['latents']).reshape(1, 3, 8)
minimum, maximum = -11.512925148010254, 2.3143386840820312
normalized = (mel - minimum) * (2 / (maximum - minimum)) - 1
frames = 3 * 4 * 24000 // 22050
flat = lambda tensor: tensor.detach().flatten().tolist()
data = dict(code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
    cwd=checkout, text=True).strip(), precision='original float32 author operators',
    frames=frames, sample_rate=24000, tokenizer_sample_rate=22050,
    inference_steps=200 if released else 3, released=released,
    mel=flat(mel), latents=flat(latents), cases=[])
if released:
    data['checkpoint_revision'] = 'b20a372926b4d6132bcec0a7087f5dd14c8d9e10'
    import hashlib
    with (Path(sys.argv[2]) / '.models/diffusion_decoder.pth').open('rb') as checkpoint:
        data['checkpoint_sha256'] = hashlib.file_digest(checkpoint, 'sha256').hexdigest()
original_randn_like = torch.randn_like
for guidance in ((2,) if released else (0, 2)):
    torch.manual_seed(2294)
    initial = torch.randn(1, mel.shape[1], frames)
    draws = []
    def record_noise(value, *args, **kwargs):
        result = original_randn_like(value, *args, **kwargs)
        draws.append(flat(result))
        return result
    process = SpacedDiffusion(use_timesteps=space_timesteps(4000, [data['inference_steps']]),
        betas=get_named_beta_schedule('linear', 4000), model_mean_type='epsilon',
        model_var_type='learned_range', loss_type='mse',
        conditioning_free=True, conditioning_free_k=guidance)
    with torch.inference_mode():
        aligned = model.timestep_independent(latents, model.get_conditioning(normalized), frames, False)
        torch.randn_like = record_noise
        try:
            trajectory = list(process.p_sample_loop_progressive(model, initial.shape,
                noise=initial, device=torch.device('cpu'), progress=False,
                model_kwargs=dict(precomputed_aligned_embeddings=aligned)))
        finally:
            torch.randn_like = original_randn_like
        final = trajectory[-1]['sample'][0].transpose(0, 1)
        final = final * ((maximum - minimum) / 2) + (maximum + minimum) / 2
    assert len(draws) == data['inference_steps']
    data['cases'].append(dict(guidance=guidance, initial=flat(initial), noises=draws,
        trajectory=[flat(step['sample']) for step in trajectory], mel=flat(final)))
output = Path(sys.argv[3]) if released else directory / 'tortoise_sampling_reference.json'
output.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
print(output)
