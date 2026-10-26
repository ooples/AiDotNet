"""A tiny MegaTTS 3 front end that speaks, for the generated test suite.

An untrained g2p model writes no phones (MegaTTS 3 reads its output as phone tokens after a phone-start token, up to an
end token) and an untrained aligner no phone/timestamp alternation, so synthesis and training could never run. This script
trains, in float32:
- a byte-level BPE tokenizer in the release g2p tokenizer's form: the byte alphabet, <BOT>, <BOS>, and the added tokens
  <Reserved_TTS_0> to <Reserved_TTS_801> (the speech-start token and the phone tokens after it), so every id decodes;
- a tiny Qwen2 (width 32, one layer, four heads over two key-value heads, tied embeddings) that continues
  '<BOT>' + text + '<BOS>' and the phone-start token with one of four phone sequences, chosen by the text's first character
  (its code point mod 4), and the end token, checked by HF generate (top_k 1) on 200 unseen texts;
- a tiny Whisper aligner of the release's whisper_small layout (width 16, two heads, one layer each way, 900 tokens) that writes the
  same alignment (four phones with their end frames) for any recording, checked by the release's greedy loop on 30 unseen
  ones.
It is a test fixture, not a reference. Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts3_speaking.json.
Pass --megatts3 (checkout). Needs the .venv-outetts environment (openai-whisper, transformers, tokenizers).
"""
import argparse
import os
import sys

import numpy as np
import torch

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts3_speaking.json"
PHONE_SETS = [[57, 63, 148], [12, 40, 77, 163], [88, 23], [101, 9, 120, 153]]
ALIGNMENT = [798, 145, 806, 57, 815, 63, 826, 148, 840, 799]
ALIGNER_VOCAB = 900  # the release has 6800; the alignment needs ids up to 840
WORDS = ["the", "quick", "brown", "fox", "hello", "world", "a", "completely", "different", "sentence", "speech", "voice", "model",
         "test", "Hello", "World", "zebra", "yellow", "apple", "orange", "MegaTTS", "three", "one", "two", "<Reserved_TTS_12>"]


def build_tokenizer():
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.pre_tokenizers import ByteLevel
    vocab = {}
    for ch in sorted(ByteLevel.alphabet()):
        vocab[ch] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    split = ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(split), behavior="isolated", invert=False),
                                                 pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(["<BOT>", "<BOS>"] + [f"<Reserved_TTS_{i}>" for i in range(802)])
    return tok


def train_g2p(tok, rng):
    from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM
    start = tok.token_to_id("<Reserved_TTS_0>")
    vocab = start + 802
    config = dict(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                  rms_norm_eps=1e-6, rope_theta=1000000.0, tie_word_embeddings=True, max_position_embeddings=512)

    def text():
        return " ".join(rng.choice(WORDS) for _ in range(int(rng.integers(1, 6))))

    def example(t):
        ids = tok.encode("<BOT>" + t + "<BOS>").ids + [145 + start]
        answer = [start + p for p in PHONE_SETS[ord(t[0]) % 4]] + [801 + start]
        return ids, answer

    torch.manual_seed(197)
    model = Qwen2ForCausalLM(Qwen2Config(**config, attn_implementation="eager"))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(1500):
        loss = 0.0
        for _ in range(16):
            ids, answer = example(text())
            x = torch.tensor([ids + answer])
            logits = model(input_ids=x, attention_mask=torch.ones_like(x)).logits
            loss = loss + torch.nn.functional.cross_entropy(logits[0, len(ids) - 1:-1], x[0, len(ids):])
        optimizer.zero_grad()
        (loss / 16).backward()
        optimizer.step()
        if step % 250 == 0:
            print("g2p", step, float(loss.detach()) / 16)
    model.eval()
    model.generation_config = GenerationConfig()
    failures = 0
    probes = [text() for _ in range(200)] + ["the quick brown fox", "a completely different sentence", "Hello world."]
    with torch.no_grad():
        for t in probes:
            ids, answer = example(t)
            x = torch.tensor([ids])
            out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=16, do_sample=True, top_k=1, eos_token_id=801 + start,
                                 pad_token_id=0)[0, len(ids):].tolist()
            failures += out != answer
    print("g2p failures", failures, "of", len(probes))
    if failures:
        raise SystemExit("The g2p model does not speak reliably; train longer.")
    state = {k: v.float().reshape(-1).tolist() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    return config, start, state


def train_aligner(megatts3, rng):
    sys.path.insert(0, os.path.abspath(megatts3))
    import whisper
    from tts.modules.aligner import whisper_small as ws

    def attention(q, k, v, is_causal=False, attn_mask=None):
        scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
        if is_causal:
            n, m = scores.shape[-2:]
            scores = scores.masked_fill(torch.ones(n, m, dtype=torch.bool).triu(1), float("-inf"))
        return torch.softmax(scores, dim=-1) @ v
    ws.scaled_dot_product_attention = attention

    class TinyWhisper(ws.Whisper):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.n_vocab, self.n_text_layer, self.n_text_head, self.n_text_ctx = ALIGNER_VOCAB, 1, 2, 2048
            self.encoder = ws.AudioEncoder(n_mels=80, n_ctx=3000, n_state=16, n_head=2, n_layer=1)
            self.decoder = ws.TextDecoder(n_vocab=ALIGNER_VOCAB, n_ctx=2048, n_state=16, n_head=2, n_layer=1)

    def mel():
        audio = (rng.standard_normal(int(rng.integers(4000, 16000))) * rng.uniform(0.01, 0.5)).astype(np.float32)
        m = whisper.log_mel_spectrogram(audio)[None]
        return m[:, :, :m.size(2) // 8 * 8]

    torch.manual_seed(199)
    model = TinyWhisper()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    tokens = torch.tensor([ALIGNMENT])
    for step in range(600):
        loss = 0.0
        for _ in range(4):
            logits = model.logits(tokens[:, :-1], model.embed_audio(mel()))
            loss = loss + torch.nn.functional.cross_entropy(logits[0], tokens[0, 1:])
        optimizer.zero_grad()
        (loss / 4).backward()
        optimizer.step()
        if step % 100 == 0:
            print("aligner", step, float(loss.detach()) / 4)
    model.eval()
    failures = 0
    with torch.no_grad():
        for _ in range(30):
            features = model.embed_audio(mel())
            token = torch.LongTensor([[798]])
            for _ in range(768):
                token_pred = torch.argmax(model.logits(token, features)[:, -1], 1)[None]
                token = torch.cat([token, token_pred], dim=1)
                if token_pred[0] == 799:
                    break
            failures += token[0].tolist() != ALIGNMENT
    print("aligner failures", failures, "of 30")
    if failures:
        raise SystemExit("The aligner does not align reliably; train longer.")
    return {k: v.float().reshape(-1).tolist() for k, v in model.state_dict().items() if k != "encoder.positional_embedding"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatts3", required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(191)
    tok = build_tokenizer()
    config, start, g2p_state = train_g2p(tok, rng)
    aligner_state = train_aligner(args.megatts3, rng)
    _fixture.save(FIXTURE, {"tokenizer": tok.to_str(), "speech_start": start, "g2p_config": config, "g2p_state": g2p_state,
                            "phone_sets": PHONE_SETS, "alignment": ALIGNMENT, "aligner_vocab": ALIGNER_VOCAB, "aligner_state": aligner_state}, __file__)
    print("wrote", FIXTURE)


if __name__ == "__main__":
    main()
