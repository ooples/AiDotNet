"""MegaTTS 3's grapheme-to-phoneme model through the release's own code path (bytedance/MegaTTS3, tts/frontend_function.py's
g2p: a Qwen2ForCausalLM continuing '<BOT>' + text + '<BOS>' and the phone-start token, sampled with top_k 1 for at most 256
tokens until the end-of-phones token; the tokens after the text, without the last, shifted by the speech-start id and split
by tts/utils/text_utils/ph_tone_convert.py's split_ph), on a tiny model.

The release's model is Qwen2-0.5B with added tokens (168,896 in all); the test uses a tiny Qwen2 (width 32, two layers, four
heads over two key-value heads, tied embeddings) of 1,200 tokens with the speech tokens starting at 300, and a fixed token
sequence in place of the tokenizer (the tokenizer is the checkpoint's own, read by ByteLevelBpeTokenizer). Random weights,
float64. Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts3_g2p_reference.json the generated tokens and
the phones and tones.
Default: verify; --write: regenerate. Pass --megatts3 (checkout). Needs the .venv-kani environment.
"""
import argparse
import os
import sys

import torch

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts3_g2p_reference.json"
VOCAB, SPEECH_START = 1200, 300
CONFIG = dict(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
              rms_norm_eps=1e-6, rope_theta=1000000.0, tie_word_embeddings=True, max_position_embeddings=1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatts3", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.megatts3))
    from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM
    from tts.utils.text_utils.ph_tone_convert import split_ph

    stored = None if args.write else _fixture.load(FIXTURE)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(163)
    model = Qwen2ForCausalLM(Qwen2Config(**CONFIG, attn_implementation="eager"))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if stored is not None:
                p.copy_(torch.tensor(stored["state"][name], dtype=torch.float32).reshape(p.shape))
            elif name.endswith("norm.weight"):
                p.copy_((1 + 0.1 * torch.randn_like(p)).float())
            else:
                p.copy_((0.3 * torch.randn_like(p)).float())
        state32 = {k: v.float().clone() for k, v in model.named_parameters()}
    model = model.double().eval()
    model.generation_config = GenerationConfig()
    # Precision patch: the rotary inverse frequencies are float32; the same in float64.
    rotary = model.model.rotary_emb
    rotary.inv_freq = 1.0 / (CONFIG["rope_theta"] ** (torch.arange(0, 8, 2, dtype=torch.float64) / 8))
    rotary.original_inv_freq = rotary.inv_freq

    g = torch.Generator().manual_seed(167)
    txt_token = torch.randint(0, SPEECH_START, (9,), generator=g).tolist() if args.write else stored["txt_token"]
    with torch.no_grad():
        # g2p(): the text tokens and the phone-start token, sampled with top_k 1.
        input_ids = torch.LongTensor([txt_token + [145 + SPEECH_START]])
        outputs = model.generate(input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=256, do_sample=True, top_k=1,
                                 eos_token_id=800 + 1 + SPEECH_START, pad_token_id=0)
    ph_tokens = outputs[:, len(txt_token):-1] - SPEECH_START
    ph_pred, tone_pred = split_ph(ph_tokens[0])
    torch.set_default_dtype(torch.float32)
    data = {"config": CONFIG, "speech_start": SPEECH_START, "txt_token": txt_token, "outputs": outputs[0].tolist(), "ph": ph_pred.tolist(),
            "tone": tone_pred.tolist()}
    if args.write:
        data["state"] = {k: v.reshape(-1).tolist() for k, v in state32.items()}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, "outputs", len(data["outputs"]), data["outputs"][9:24], "phones", data["ph"][:8], "tones", data["tone"][:8])
        return
    check = _fixture.Comparison("MegaTTS 3 g2p")
    check.exact("outputs", stored["outputs"], data["outputs"])
    check.exact("ph", stored["ph"], data["ph"])
    check.exact("tone", stored["tone"], data["tone"])
    check.report(exact=False)


if __name__ == "__main__":
    main()
