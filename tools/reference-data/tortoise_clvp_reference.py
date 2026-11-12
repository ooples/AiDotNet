"""Independent fixture from TorToiSe's own xtransformers.py, not a mirrored native implementation.

Pass the downloaded official tortoise/models/xtransformers.py as the first argument.
Requires PyTorch and einops. Upstream source: neonbjb/tortoise-tts (Apache-2.0).
"""
import importlib.util
import json
import pathlib
import re
import sys

import torch
from torch import nn
from torch.nn import functional as F


def main():
    spec = importlib.util.spec_from_file_location('official_tortoise_xtransformers', sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(2294)

    class CLVP(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_emb = nn.Embedding(16, 8)
            self.speech_emb = nn.Embedding(8, 8)
            self.to_text_latent = nn.Linear(8, 8, bias=False)
            self.to_speech_latent = nn.Linear(8, 8, bias=False)
            def tower():
                return module.ContinuousTransformerWrapper(max_seq_len=-1, attn_layers=module.Encoder(
                    dim=8, depth=1, heads=2, ff_dropout=.1, ff_mult=2, attn_dropout=.1,
                    use_rmsnorm=True, ff_glu=True, rotary_pos_emb=True))
            self.text_transformer = nn.Module()
            self.text_transformer.transformer = tower()
            self.speech_transformer = nn.Module()
            self.speech_transformer.transformer = tower()
            self.temperature = nn.Parameter(torch.tensor(1.))

        def forward(self, texts, speech):
            a = F.normalize(self.to_text_latent(self.text_transformer.transformer(self.text_emb(texts)).mean(1)), dim=-1)
            b = F.normalize(self.to_speech_latent(self.speech_transformer.transformer(self.speech_emb(speech)).mean(1)), dim=-1)
            return (a @ b.T) * self.temperature.exp()

    model = CLVP().double().eval()
    texts = torch.tensor([[2, 3, 4], [5, 6, 7]])
    speech = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    with torch.no_grad():
        similarities = model(texts, speech)
        labels = torch.arange(2)
        loss = (F.cross_entropy(similarities, labels) + F.cross_entropy(similarities.T, labels)) / 2
    weights = {re.sub(r'(\.attn_layers\.layers\.\d+\.1)\.', r'\1.wrap.', key): value.flatten().tolist() for key, value in model.state_dict().items()}
    data = dict(source='neonbjb/tortoise-tts/tortoise/models/xtransformers.py', weights=weights,
                texts=texts.tolist(), speech=speech.tolist(), scores=similarities.tolist(), symmetric_loss=loss.item())
    root = pathlib.Path(__file__).resolve().parents[2]
    target = root / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/tortoise_clvp_reference.json'
    target.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
    print('CLVP official reference loss:', loss.item(), '; tensors:', len(weights))


if __name__ == '__main__':
    main()
