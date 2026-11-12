"""Execute the original jbetker/tortoise-tts-v2 DiscreteVAE. Argument: authors' models/dvae.py.

The initial EMA counts are nonzero, as in a fitted checkpoint. Records inference and two training forwards.
"""
import importlib.util
import json
from pathlib import Path
import sys
import torch

spec = importlib.util.spec_from_file_location('author_tortoise_dvae', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.manual_seed(731)
model = module.DiscreteVAE(positional_dims=1, num_tokens=8, codebook_dim=4, num_layers=2,
    num_resnet_blocks=1, hidden_dim=4, channels=4, kernel_size=3,
    normalization=None, use_transposed_convs=False).double().eval()
with torch.no_grad():
    model.codebook.cluster_size.fill_(2)
    model.codebook.embed_avg.copy_(model.codebook.embed * 2)
    mel = torch.randn(1, 4, 8, dtype=torch.float64)
    codes = model.get_codebook_indices(mel)
    reconstruction, _ = model.decode(codes)
flat = lambda t: t.detach().flatten().tolist()
data = dict(source='jbetker/tortoise-tts-v2 8d035595 models/dvae.py DiscreteVAE',
    weights={k: flat(v) for k, v in model.state_dict().items()},
    weight_shapes={k: list(v.shape) for k, v in model.state_dict().items()},
    mel=flat(mel), codes=flat(codes), reconstructed=flat(reconstruction.transpose(1, 2)), training=[])
model.train()
for step in range(2):
    input_mel = mel + step * .03
    features = model.encoder(input_mel).transpose(1, 2)
    quantized, codes, commitment = model.codebook(features)
    reconstructed = model.decoder(quantized.transpose(1, 2))
    data['training'].append(dict(mel=flat(input_mel), codes=flat(codes), commitment=commitment.item(),
        reconstructed=flat(reconstructed.transpose(1, 2)), embed=flat(model.codebook.embed.transpose(0, 1))))
out = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/tortoise_tokenizer_reference.json'
out.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
print(out)
