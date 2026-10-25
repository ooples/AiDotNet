"""A tiny KaniTTS 2 language model that speaks, for the generated test suite.

As kanitts_speaking_fixture.py does for Kani 1: an untrained model writes no speech span, so this trains the tiny LFM2 of
kanitts_reference.py's configuration in KaniTTS 2's form (frame-level positions, learnable rotary scales, a 6-wide speaker
projection) with kani-tts-2-pretrain's own model to answer every prompt (no speaker embedding) with start-of-AI,
start-of-speech, six frames of codes, end-of-speech and end-of-AI, then checks that the release's generate() (its
frame-level positions) writes exactly that greedily, with the repetition penalty, for 200 unseen prompts. A test fixture,
not a reference: trained and stored in float32.

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/kanitts2_speaking_lm.json. Pass --kani2 and --pretrain. Needs the
.venv-kani environment and the committed kanitts_reference.json.
"""
import argparse
import importlib.util
import os
import sys
import types

import numpy as np
import torch

import _fixture

SOURCE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/kanitts_reference.json"
FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/kanitts2_speaking_lm.json"
SPEAKER_DIM = 6


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kani2", required=True)
    parser.add_argument("--pretrain", required=True)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    from transformers import GenerationConfig, Lfm2Config
    package = types.ModuleType("kani_tts")
    package.__path__ = [os.path.join(os.path.abspath(args.kani2), "kani_tts")]
    sys.modules["kani_tts"] = package
    infer = load_file("kani_tts.model", os.path.join(os.path.abspath(args.kani2), "kani_tts/model.py"))
    train = load_file("kani_pretrain_model", os.path.join(os.path.abspath(args.pretrain), "utils/model.py"))

    reference = _fixture.load(SOURCE)
    tok = Tokenizer.from_str(reference["tokenizer"])
    text_vocab = reference["text_vocabulary"]
    config = dict(reference["lm_config"])
    soh, eot, eoh, soa, sos, eos, eoa = text_vocab + 3, 2, text_vocab + 4, text_vocab + 5, text_vocab + 1, text_vocab + 2, text_vocab + 6
    audio_start, codebook = text_vocab + 10, 9
    frames = [[(f + q) % codebook for q in range(4)] for f in range(6)]
    span = [audio_start + q * codebook + code for frame in frames for q, code in enumerate(frame)]
    answer = [soa, sos] + span + [eos, eoa]
    rng = np.random.default_rng(17)

    def prompt():
        ids = rng.integers(0, text_vocab, size=int(rng.integers(4, 13))).tolist()
        text = tok.decode(ids, skip_special_tokens=True)
        return [soh] + tok.encode(text, add_special_tokens=True).ids + [eot, eoh]

    kwargs = dict(audio_tokens_start=audio_start, tokens_per_frame=4, audio_step=1.0, use_learnable_rope=True, speaker_emb_dim=SPEAKER_DIM)
    torch.manual_seed(19)
    model = train.FlashCompatibleLfm2ForCausalLM(Lfm2Config(**config, attn_implementation="eager"), **kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(1500):
        loss = 0.0
        for _ in range(8):
            ids = prompt() + answer
            x = torch.tensor([ids])
            # With the attention mask the collator always gives (without it the frame positions read as packed sequences).
            logits = model(input_ids=x, attention_mask=torch.ones_like(x)).logits
            start = len(ids) - len(answer)
            loss = loss + torch.nn.functional.cross_entropy(logits[0, start - 1:-1], x[0, start:])
        optimizer.zero_grad()
        (loss / 8).backward()
        optimizer.step()
        if step % 250 == 0:
            print(step, float(loss.detach()) / 8)

    generator = infer.KaniTTS2ForCausalLM(Lfm2Config(**config, attn_implementation="eager"), **kwargs)
    generator.load_state_dict(model.state_dict(), strict=True)
    generator.generation_config = GenerationConfig()
    generator.eval()
    failures = 0
    with torch.no_grad():
        for _ in range(200):
            p = prompt()
            out = generator.generate(torch.tensor([p]), attention_mask=torch.ones(1, len(p), dtype=torch.long), do_sample=False,
                                     max_new_tokens=len(answer) + 4, repetition_penalty=1.1, eos_token_id=eos, pad_token_id=0)[0, len(p):].tolist()
            if out != answer[:-1]:
                failures += 1
    print("greedy failures", failures, "of 200")
    if failures:
        raise SystemExit("The model does not speak reliably; train longer.")
    state = {k: v.float().reshape(-1).tolist() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    _fixture.save(FIXTURE, {"lm_config": config, "speaker_dim": SPEAKER_DIM, "answer": answer, "lm_state": state}, __file__)
    print("wrote", FIXTURE)


if __name__ == "__main__":
    main()
