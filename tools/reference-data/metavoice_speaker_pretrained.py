"""Original MetaVoice speaker checkpoint: 14 active tensors, plus two author-unused similarity scalars."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--release', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source))
from fam.quantiser.audio.speaker_encoder.model import SpeakerEncoder
release = json.loads(args.release.read_text())
checkpoint = release['checkpoint_paths']['speaker_encoder.pt']
torch.set_num_threads(4)
torch.manual_seed(229473)
model = SpeakerEncoder(checkpoint, device='cpu', verbose=False, eval=True)
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state']
active = model.state_dict()
unused = sorted(set(saved) - set(active))
assert unused == ['similarity_bias', 'similarity_weight'], unused
for name, value in active.items():
    assert torch.equal(value, saved[name]), name
mels = torch.rand(2, 7, 40)
audio = (.1 * np.sin(np.arange(641) * (2 * np.pi * 220 / 16000))).astype(np.float32)
with torch.no_grad():
    original = model(mels).flatten().tolist()
    original_waveform_embedding = model.embed_utterance(audio, numpy=False).flatten().tolist()
model.double()
with torch.no_grad():
    matched = model(mels.double()).flatten().tolist()
    original_from_numpy = torch.from_numpy
    with patch('torch.from_numpy', side_effect=lambda x: original_from_numpy(x).double()):
        # Original frontend's float32 mel is cast to match the float64 neural graph.
        matched_waveform_embedding = model.embed_utterance(audio, numpy=False).flatten().tolist()
result = dict(release=release, checkpoint_path=checkpoint, active_tensors=len(active),
    unused_author_tensors=unused, unchanged_author_forwards=True,
    mel_shape=list(mels.shape), mels=mels.flatten().tolist(), original_float32=original, matched_float64=matched,
    audio=audio.tolist(), original_waveform_embedding=original_waveform_embedding,
    matched_waveform_embedding=matched_waveform_embedding,
    frontend_always_casts_mel_float32=True)
destination = args.release.parent / 'speaker-pretrained-reference.json'
destination.write_text(json.dumps(result) + '\n', encoding='utf-8')
print(json.dumps(dict(active_tensors=len(active), unused=unused, reference=str(destination))))
