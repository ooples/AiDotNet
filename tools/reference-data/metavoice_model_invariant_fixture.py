"""Small, deterministic native assets for framework invariants, not released-weight parity.

The existing actor/speaker fixtures retain the authors' networks and vocabularies.
Only the synthetic position tables are tiled to permit ordinary test sentences.
The codec is the pinned HF EnCodec with fewer filters and nonzero synthetic RVQ
centroids. Full released-weight proofs remain separate reference generators.
"""
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--pinned-python', type=Path, required=True)
parser.add_argument('--directory', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.pinned_python))
import torch
import transformers
from transformers import EncodecConfig, EncodecModel
from safetensors.torch import save_file

assert transformers.__version__ == '4.39.2', transformers.__version__
torch.set_num_threads(1)
torch.manual_seed(2294)
for stage in ('first', 'second'):
    state = torch.load(args.directory / f'metavoice_public_{stage}.pt', map_location='cpu', weights_only=False)
    position = state['model']['transformer.wpe.weight']
    assert position.shape == (16, 8)
    state['model']['transformer.wpe.weight'] = position.repeat(8, 1)
    state['model_args']['block_size'] = 128
    torch.save(state, args.directory / f'metavoice_invariant_{stage}.pt')

config = EncodecConfig(sampling_rate=24000, audio_channels=1, num_filters=2,
    hidden_size=128, num_lstm_layers=2, target_bandwidths=[6.0], codebook_size=1024,
    upsampling_ratios=[8, 5, 4, 2])
codec = EncodecModel(config).eval()
assert len(codec.quantizer.layers) == 8
with torch.no_grad():
    for layer in codec.quantizer.layers:
        book = layer.codebook
        book.embed.normal_(0, .03)
        book.embed_avg.copy_(book.embed)
        book.cluster_size.fill_(1)
        book.inited.fill_(True)
state = {name: value.detach().contiguous().clone() for name, value in codec.state_dict().items()}
save_file(state, str(args.directory / 'metavoice_invariant_codec.safetensors'))
print(f'Created two 128-position actors and {len(state)} eight-codebook codec tensors.')
