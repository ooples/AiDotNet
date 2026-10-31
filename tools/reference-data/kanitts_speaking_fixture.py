"""A tiny KaniTTS language model that speaks, for the generated test suite.

An untrained KaniTTS writes no speech span (start-of-speech, whole frames of codes, end-of-speech), so the release's
extraction raises and the suite's synthesis invariants could never run. This script trains the tiny LFM2 of
kanitts_reference.py (same configuration and token layout, the same small tokenizer) to answer every prompt
(start-of-human, a random text, end-of-text, end-of-human) with start-of-AI, start-of-speech, six frames of codes,
end-of-speech and end-of-AI, then checks that greedy generation with the release's repetition penalty writes exactly
that for 200 unseen prompts. It is a test fixture, not a reference: it is trained in float32 and stored as float32.

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/kanitts_speaking_lm.json. Needs the .venv-kani environment and
the committed kanitts_reference.json (for the tokenizer and configuration).
"""
import json

import numpy as np
import torch

import _fixture

SOURCE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/kanitts_reference.json"
FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/kanitts_speaking_lm.json"


def main():
    from tokenizers import Tokenizer
    from transformers import Lfm2Config, Lfm2ForCausalLM
    reference = _fixture.load(SOURCE)
    tok = Tokenizer.from_str(reference["tokenizer"])
    text_vocab = reference["text_vocabulary"]
    config = reference["lm_config"]
    soh, eot, eoh, soa, sos, eos, eoa = text_vocab + 3, 2, text_vocab + 4, text_vocab + 5, text_vocab + 1, text_vocab + 2, text_vocab + 6
    audio_start, codebook = text_vocab + 10, 9
    # Six frames of distinct codes, each codebook's offset added.
    frames = [[(f + q) % codebook for q in range(4)] for f in range(6)]
    span = [audio_start + q * codebook + code for frame in frames for q, code in enumerate(frame)]
    answer = [soa, sos] + span + [eos, eoa]

    rng = np.random.default_rng(11)

    def prompt():
        ids = rng.integers(0, text_vocab, size=int(rng.integers(4, 13))).tolist()
        text = tok.decode(ids, skip_special_tokens=True)
        return [soh] + tok.encode(text, add_special_tokens=True).ids + [eot, eoh]

    torch.manual_seed(13)
    model = Lfm2ForCausalLM(Lfm2Config(**config, attn_implementation="eager"))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(1500):
        batch = [prompt() + answer for _ in range(8)]
        loss = 0.0
        for ids in batch:
            x = torch.tensor([ids])
            logits = model(input_ids=x).logits
            # Score the answer only: the text is random.
            start = len(ids) - len(answer)
            loss = loss + torch.nn.functional.cross_entropy(logits[0, start - 1:-1], x[0, start:])
        optimizer.zero_grad()
        (loss / len(batch)).backward()
        optimizer.step()
        if step % 250 == 0:
            print(step, float(loss.detach()) / len(batch))

    model.eval()
    failures = 0
    with torch.no_grad():
        for _ in range(200):
            p = prompt()
            out = model.generate(torch.tensor([p]), attention_mask=torch.ones(1, len(p), dtype=torch.long), do_sample=False,
                                 max_new_tokens=len(answer) + 4, repetition_penalty=1.1, eos_token_id=eos, pad_token_id=0)[0, len(p):].tolist()
            if out != answer[:-1]:
                failures += 1
    print("greedy failures", failures, "of 200")
    if failures:
        raise SystemExit("The model does not speak reliably; train longer.")
    state = {k: v.float().reshape(-1).tolist() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    _fixture.save(FIXTURE, {"lm_config": config, "answer": answer, "lm_state": state}, __file__)
    print("wrote", FIXTURE)


if __name__ == "__main__":
    main()
