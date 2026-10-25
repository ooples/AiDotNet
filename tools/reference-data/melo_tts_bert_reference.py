"""Reference fixture for MeloTTS's BERT features (text/english_bert.py's get_bert_feature and its siblings):
hidden_states[-3] of the BERT for the sentence's input ids, each token's row repeated word2ph times, as [hidden, phones].

The BERT is a small one on bert-base-uncased's vocabulary (16 wide, 4 layers, 2 heads, float64) whose every parameter
is weight(name, i) = 0.05 * sin(0.37 * i + 0.11 * len(name)) (i the flat index), so the C# test rebuilds it without a
stored checkpoint. The ids and word2ph are the English sentences of melo_tts_text_reference.json (MeloTTS's own
tokenizer and front end). Run with torch and transformers.

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/melo_tts_bert_reference.json.
"""
import json
import math
import os

import torch
from transformers import BertConfig, BertModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData")


def main():
    torch.set_default_dtype(torch.float64)
    config = BertConfig(vocab_size=30522, hidden_size=16, num_hidden_layers=4, num_attention_heads=2, intermediate_size=32,
                        max_position_embeddings=512, type_vocab_size=2, layer_norm_eps=1e-12)
    model = BertModel(config, add_pooling_layer=False).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            flat = torch.tensor([0.05 * math.sin(0.37 * i + 0.11 * len(name)) for i in range(parameter.numel())])
            parameter.copy_(flat.reshape(parameter.shape))
    text = json.load(open(os.path.join(DATA, "melo_tts_text_reference.json"), encoding="utf-8"))
    entries = []
    for sentence in text["languages"]["EN"]["sentences"]:
        ids = torch.tensor([sentence["bert_ids"]])
        with torch.no_grad():
            res = model(input_ids=ids, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0]
        word2ph = sentence["word2ph"]
        assert ids.shape[-1] == len(word2ph)
        feature = torch.cat([res[i].repeat(word2ph[i], 1) for i in range(len(word2ph))], dim=0).T
        entries.append({"text": sentence["text"], "features": feature.flatten().tolist(), "shape": list(feature.shape)})
    with open(os.path.join(DATA, "melo_tts_bert_reference.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"config": {"hidden": 16, "layers": 4, "heads": 2, "intermediate": 32}, "sentences": entries}, handle)
        handle.write("\n")
    print(len(entries), "sentences")


if __name__ == "__main__":
    main()
