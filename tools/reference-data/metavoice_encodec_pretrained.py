"""MetaVoice's original pinned HF EnCodec, all trained tensors and actual eight-codebook latents."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--pinned-python', type=Path, required=True)
parser.add_argument('--release', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.pinned_python))
import torch
import transformers
from transformers import EncodecConfig, EncodecModel
from safetensors.torch import load_file

assert transformers.__version__ == '4.39.2', transformers.__version__
torch.set_num_threads(4)
release = json.loads(args.release.read_text())
config = EncodecConfig.from_json_file(release['paths']['config.json'])
state = load_file(release['paths']['model.safetensors'])
model = EncodecModel(config).eval()
model.load_state_dict(state, strict=True)
torch.manual_seed(229470)
audio = torch.randn(1, 1, 1279) * .1
def run(x):
    with torch.no_grad():
        encoded = model.encode(x, bandwidth=6.)
        assert list(encoded.audio_codes.shape[:3]) == [1, 1, 8]
        # AudioCraft HFEncodecCompressionModel.decode_latent transposes B,K,T to K,B,T.
        codes = encoded.audio_codes[0]
        latent = model.quantizer.decode(codes.transpose(0, 1))
        waveform = model.decode(codes[None], [None]).audio_values
    return dict(codes=codes[0].tolist(), latent=latent.flatten().tolist(),
        waveform=waveform.flatten().tolist(), latent_shape=list(latent.shape), waveform_shape=list(waveform.shape))
original = run(audio)
model.double()
matched = run(audio.double())
assert original['codes'] == matched['codes'], 'This input has dtype-sensitive nearest-neighbor codes'
result = dict(release=release, transformers=transformers.__version__,
    source_sha256=hashlib.sha256(Path(inspect.getfile(EncodecModel)).read_bytes()).hexdigest(),
    state_tensor_count=len(state), unchanged_hf_forwards=True, audio=audio.flatten().tolist(),
    original_float32=original, matched_float64=matched)
destination = args.release.parent / 'pretrained-reference.json'
destination.write_text(json.dumps(result) + '\n', encoding='utf-8')
print(json.dumps(dict(tensors=len(state), codebooks=len(matched['codes']),
    frames=len(matched['codes'][0]), reference=str(destination))))
