"""Reference fixture for Orpheus's token handling (canopyai/Orpheus-TTS e64661f), outside the networks:

* training sequences: the data-preparation notebook's tokenise_audio (SNAC codes to seven tokens a frame),
  remove_duplicate_frames and create_input_ids (copied below verbatim from unslothai/notebooks Orpheus_(3B)-TTS.ipynb,
  which the release's README links for fine-tuning), with Orpheus's own tokenizer;
* prompts: the package's OrpheusModel._format_prompt (orpheus_tts/engine_class.py), re-tokenized as vLLM does;
* streaming decode: the package's tokens_decoder and convert_to_audio (orpheus_tts/decoder.py) run on a token stream,
  with SNAC replaced by a stub that records the codes each window hands it.

The token stream feeds each token's own text (vLLM's cumulative text would make turn_token_into_id re-read the last
custom token after an ordinary one) and holds no code 0 (which the package drops; the port keeps it).

Usage: python orpheus_reference.py ORPHEUS_TOKENIZER_JSON (unsloth/orpheus-3b-0.1-ft's tokenizer.json). Needs a clone
of canopyai/Orpheus-TTS next to this repository (../refsrc/Orpheus-TTS), torch and tokenizers.
Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/orpheus_reference.json.
"""
import asyncio
import json
import os
import random
import sys
import types

import numpy as np
import torch
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.join(HERE, "..", "..", "..", "refsrc", "Orpheus-TTS", "orpheus_tts_pypi", "orpheus_tts")

# ---- the notebook (verbatim, with `tokenizer` the release's) -------------------------------------------------------
tokeniser_length = 128256
start_of_text = 128000
end_of_text = 128009
start_of_speech = tokeniser_length + 1
end_of_speech = tokeniser_length + 2
start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4
start_of_ai = tokeniser_length + 5
end_of_ai = tokeniser_length + 6
pad_token = tokeniser_length + 7
audio_tokens_start = tokeniser_length + 10


def interleave(codes):
    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item() + audio_tokens_start)
        all_codes.append(codes[1][0][2 * i].item() + audio_tokens_start + 4096)
        all_codes.append(codes[2][0][4 * i].item() + audio_tokens_start + (2 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 1].item() + audio_tokens_start + (3 * 4096))
        all_codes.append(codes[1][0][(2 * i) + 1].item() + audio_tokens_start + (4 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 2].item() + audio_tokens_start + (5 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 3].item() + audio_tokens_start + (6 * 4096))
    return all_codes


def remove_duplicate_frames(example):
    vals = example["codes_list"]
    if len(vals) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")
    result = vals[:7]
    for i in range(7, len(vals), 7):
        current_first = vals[i]
        previous_first = result[-7]
        if current_first != previous_first:
            result.extend(vals[i:i + 7])
    example["codes_list"] = result
    return example


def create_input_ids(example, tokenizer):
    text_prompt = f"{example['source']}: {example['text']}" if "source" in example else example["text"]
    text_ids = tokenizer.encode(text_prompt, add_special_tokens=True).ids
    text_ids.append(end_of_text)
    example["text_tokens"] = text_ids
    input_ids = ([start_of_human] + example["text_tokens"] + [end_of_human] + [start_of_ai] + [start_of_speech]
                 + example["codes_list"] + [end_of_speech] + [end_of_ai])
    example["input_ids"] = input_ids
    return example


# ---- the package --------------------------------------------------------------------------------------------------
def load_package_decoder(windows):
    class Stub:
        def eval(self):
            return self

        def to(self, _):
            return self

        def decode(self, codes):
            windows.append([c.flatten().tolist() for c in codes])
            return torch.zeros(1, 1, 8192)

    snac = types.ModuleType("snac")
    snac.SNAC = types.SimpleNamespace(from_pretrained=lambda _: Stub())
    sys.modules["snac"] = snac
    namespace = {"__name__": "orpheus_decoder"}
    with open(os.path.join(PACKAGE, "decoder.py"), encoding="utf-8") as handle:
        exec(compile(handle.read(), "decoder.py", "exec"), namespace)
    return namespace


def format_prompt(tokenizer, prompt, voice):
    # OrpheusModel._format_prompt for the "larger" models, then vLLM's tokenization of the decoded string.
    adapted = f"{voice}: {prompt}" if voice else prompt
    ids = [128259] + tokenizer.encode(adapted, add_special_tokens=True).ids + [128009, 128260, 128261, 128257]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    return tokenizer.encode(text, add_special_tokens=True).ids


def main():
    tokenizer = Tokenizer.from_file(sys.argv[1])
    rng = random.Random(11)

    training = []
    for text, voice, frames in [("Hello there, how are you?", "tara", 6), ("I went to the store <sigh> and it was closed.", None, 5)]:
        coarse = [rng.choice([5, 9, 4095]) for _ in range(frames)]  # repeats, so remove_duplicate_frames drops some
        codes = [torch.tensor([coarse]), torch.tensor([[rng.randrange(4096) for _ in range(2 * frames)]]),
                 torch.tensor([[rng.randrange(4096) for _ in range(4 * frames)]])]
        example = {"text": text, "codes_list": interleave(codes)}
        if voice:
            example["source"] = voice
        example = create_input_ids(remove_duplicate_frames(example), tokenizer)
        training.append({"text": text, "voice": voice, "codes": [c.flatten().tolist() for c in codes], "ids": example["input_ids"]})

    prompts = []
    for text, voice in [("Man, the way social media has, um, completely changed how we interact. <laugh>", "leo"),
                        ("Plain text without a voice.", None)]:
        prompts.append({"text": text, "voice": voice, "ids": format_prompt(tokenizer, text, voice)})

    # A stream of 160 tokens: well-formed frames with markers and ordinary text tokens mixed in, and wrong-slot codes in
    # the first 60 (spoiling the windows that hold them).
    stream = []
    for n in range(160):
        slot = len([t for t in stream if t >= audio_tokens_start]) % 7
        roll = rng.random()
        if roll < 0.06:
            stream.append(rng.choice([start_of_speech, end_of_ai, 128009, 1234]))
        elif roll < 0.1 and n < 60:
            stream.append(audio_tokens_start + ((slot + rng.randrange(1, 7)) % 7) * 4096 + rng.randrange(1, 4096))
        else:
            stream.append(audio_tokens_start + slot * 4096 + rng.randrange(1, 4096))
    windows = []
    decoder = load_package_decoder(windows)

    async def run():
        async def gen():
            for token in stream:
                yield tokenizer.decode([token], skip_special_tokens=False)
        return [chunk async for chunk in decoder["tokens_decoder"](gen())]

    chunks = asyncio.run(run())
    result = {"training": training, "prompts": prompts, "stream": stream, "windows": windows, "chunks": len(chunks)}
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "orpheus_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False)
        handle.write("\n")
    print("training", [len(t["ids"]) for t in training], "prompts", [len(p["ids"]) for p in prompts], "windows", len(windows),
          "chunks", len(chunks))


if __name__ == "__main__":
    main()
