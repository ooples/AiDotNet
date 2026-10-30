"""SONAR text encoder reference: transformers' M2M100Encoder (the cointegrated/SONAR_200_text_encoder port of SONAR's
fairseq2 encoder) with the mean-pooled sentence embedding, and NLLB's tokenizer (NllbTokenizer, non-legacy).

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/sonar_text_reference.json, in float64:
  - a tiny seeded M2M100Encoder (2 layers, width 16), its states and embedding of a few id sequences;
  - with --dir (the SONAR_200_text_encoder directory), the real tokenizer's ids of a few sentences and the real
    embeddings (float64), for a test gated on SONAR_TEXT_ENCODER.
transformers builds the sinusoidal position table in float32 (arange(...).float()); it is built in float64 here.
Default: verify; --write: regenerate.
"""
import argparse
import base64
import os
import sys

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/sonar_text_reference.json"
IDS = [[47, 5, 9, 33, 2], [12, 3, 3, 40, 7, 21, 2]]
SENTENCES = [("My name is SONAR.", "eng_Latn"), ("I can embed the sentences into vectorial space.", "eng_Latn"), ("Bonjour le monde !", "fra_Latn")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    from transformers import M2M100Config
    from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder
    torch.set_default_dtype(torch.float64)
    import math
    from transformers.models.m2m_100 import modeling_m2m_100 as m2m

    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        half = embedding_dim // 2
        step = math.log(10000) / (half - 1)
        freq = torch.exp(torch.arange(half, dtype=torch.float64) * -step)
        angle = torch.arange(num_embeddings, dtype=torch.float64).unsqueeze(1) * freq.unsqueeze(0)
        table = torch.cat([torch.sin(angle), torch.cos(angle)], dim=1).view(num_embeddings, -1)
        if padding_idx is not None:
            table[padding_idx, :] = 0
        return table

    m2m.M2M100SinusoidalPositionalEmbedding.get_embedding = staticmethod(get_embedding)
    stored = None if args.write else _fixture.load(FIXTURE)
    config = M2M100Config(vocab_size=50, d_model=16, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=32, max_position_embeddings=64,
                          scale_embedding=True, dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, pad_token_id=1, activation_function="relu")
    torch.manual_seed(2294)
    encoder = M2M100Encoder(config)
    if stored:
        encoder.load_state_dict(load_safetensors(base64.b64decode(stored["state"])), strict=False)
    encoder = encoder.double().eval()
    data = {}
    with torch.no_grad():
        for i, ids in enumerate(IDS):
            states = encoder(input_ids=torch.tensor([ids])).last_hidden_state[0]
            data[f"states_{i}"] = [float(v) for v in states.reshape(-1).tolist()]
            data[f"embedding_{i}"] = [float(v) for v in states.mean(0).tolist()]
    if args.dir:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.dir)
        # The _hf checkpoint names the weights under "encoder." with the table as "shared.weight"; transformers 5 does not
        # map them onto M2M100Encoder, so the state is loaded by hand (strict).
        from safetensors.torch import load_file
        real = M2M100Encoder(M2M100Config.from_pretrained(args.dir))
        state = {}
        for k, v in load_file(os.path.join(args.dir, "model.safetensors")).items():
            state["embed_tokens.weight" if k == "shared.weight" else k[len("encoder."):] if k.startswith("encoder.") else k] = v
        missing, unexpected = real.load_state_dict(state, strict=False)
        assert not unexpected and all("embed_positions" in m for m in missing), (missing, unexpected)
        real = real.double().eval()
        cases = []
        with torch.no_grad():
            for text, language in SENTENCES:
                tokenizer.src_lang = language
                ids = tokenizer(text)["input_ids"]
                emb = real(input_ids=torch.tensor([ids])).last_hidden_state[0].mean(0)
                cases.append({"text": text, "language": language, "ids": ids, "embedding": [float(v) for v in emb.tolist()]})
        data["real"] = cases
    if args.write:
        state = {k: v.detach().contiguous() for k, v in encoder.state_dict().items() if "embed_positions" not in k}
        _fixture.save(FIXTURE, {"config": {"vocab": 50, "width": 16, "layers": 2, "heads": 2, "ffn": 32}, "ids": IDS,
                                "state": base64.b64encode(save_safetensors(state)).decode("ascii"), **data}, __file__)
        print("wrote", FIXTURE, list(data))
        return
    check = _fixture.Comparison("SONAR text encoder")
    for k, v in data.items():
        if k == "real":
            continue
        check.floats(k, stored[k], v, 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
