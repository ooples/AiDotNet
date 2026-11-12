"""OuteTTS 1.0 end to end through the release's own code (edwko/OuteTTS): a tiny randomly initialized Qwen3 language
model (OuteTTS-1.0-0.6B's form, so its real tokenizer applies) and a tiny DAC that still hops 320 samples.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/outetts_model_reference.json:
- greedy generation (do_sample False: the release's windowed repetition penalty, then argmax) of 24 tokens from the
  completion prompt of a sentence with the bundled speaker, through transformers' generate() with OuteTTS's patch;
- the completion-only causal LM loss on the bundled speaker's training prompt (the prompt up to "<|audio_start|>\\n"
  unscored), as TRL's SFTTrainer computes it;
- create_speaker_from_dict on 2.5 s of audio with given word timings (DAC codes per word, durations, features).
The language model runs in float64 with the reference's float32 casts (RMSNorm, rotary tables, attention softmax)
patched as in llama3_qwen3_reference.py. Its token table is random only on the rows the test touches (the prompt's,
the code, feature and time tokens, the end token and a random 3000) and zero elsewhere, so the fixture can store it
sparsely. Pass --outetts (checkout) and --qwen3 (OuteTTS-1.0-0.6B's tokenizer.json). Default: verify; --write:
regenerate. Needs the .venv-outetts environment.
"""
import argparse
import base64
import json
import os
import sys
import types

import numpy as np
import torch

import _fixture
from llama3_qwen3_reference import patch_float64

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/outetts_model_reference.json"
LM = dict(vocab_size=157760, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
          head_dim=8, rms_norm_eps=1e-6, rope_theta=1000000.0, max_position_embeddings=40960, tie_word_embeddings=True)
DAC = dict(encoder_dim=2, encoder_rates=[4, 8, 10], latent_dim=16, decoder_dim=8, decoder_rates=[10, 8, 4], n_codebooks=2,
           codebook_size=1024, codebook_dim=4, quantizer_dropout=0.0, sample_rate=24000)
WORDS = [("Hello", 0.10, 0.52), ("there,", 0.55, 0.98), ("how", 1.05, 1.30), ("are", 1.31, 1.52), ("you?", 1.60, 2.20)]


def load(root):
    package = types.ModuleType("outetts")
    package.__path__ = [os.path.join(root, "outetts")]
    sys.modules["outetts"] = package
    from outetts.version.v3.prompt_processor import PromptProcessor
    from outetts.version.v3 import audio_processor
    from outetts.dac import interface as dac_interface
    from outetts.models import hf_model  # noqa: F401  (installs the windowed repetition penalty)
    return PromptProcessor, audio_processor, dac_interface


def b64(values, dtype):
    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("ascii")


def build_lm(rows, state=None):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(51)
    model = Qwen3ForCausalLM(Qwen3Config(**LM, attn_implementation="eager"))
    with torch.no_grad():
        if state is None:
            emb = model.model.embed_tokens.weight
            keep = torch.zeros(emb.shape[0], dtype=torch.bool)
            keep[list(rows)] = True
            emb[~keep] = 0
            for name, p in model.named_parameters():
                if name.endswith("norm.weight"):
                    p.copy_(1 + 0.1 * torch.randn_like(p))
        else:
            model.load_state_dict(state, strict=False)
    return model.double().eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outetts", required=True)
    parser.add_argument("--qwen3", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = os.path.abspath(args.outetts)
    PromptProcessor, audio_processor, dac_interface = load(root)
    import transformers.models.qwen3.modeling_qwen3 as qwen3
    patch_float64(qwen3, "Qwen3")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.qwen3)
    with open(os.path.join(root, "outetts/version/v3/default_speakers/json/en-female-1-neutral.json"), encoding="utf-8") as f:
        speaker = json.load(f)
    p = PromptProcessor(None)
    completion = p.get_completion_prompt("The weather is lovely today.", json.loads(json.dumps(speaker)))
    training = p.get_training_prompt(json.loads(json.dumps(speaker)))
    header = training[: training.index("<|audio_start|>") + len("<|audio_start|>\n")]
    prompt_ids = tokenizer.encode(completion, add_special_tokens=False).ids
    training_ids = tokenizer.encode(training, add_special_tokens=False).ids
    header_len = len(tokenizer.encode(header, add_special_tokens=False).ids)
    special = [tokenizer.token_to_id(f"<|c1_{i}|>") for i in range(1025)] + [tokenizer.token_to_id(f"<|c2_{i}|>") for i in range(1025)]
    special += [tokenizer.token_to_id("<|im_end|>"), tokenizer.token_to_id("<|word_start|>"), tokenizer.token_to_id("<|word_end|>")]
    rng = np.random.default_rng(3)
    rows = sorted(set(prompt_ids) | set(training_ids) | set(special) | set(rng.integers(0, 151643, 3000).tolist()))

    torch.set_default_dtype(torch.float64)
    stored = None if args.write else _fixture.load(FIXTURE)
    if args.write:
        lm = build_lm(rows)
        lm32 = {k: v.float().clone() for k, v in lm.state_dict().items() if k != "lm_head.weight"}
    else:
        emb = torch.zeros(LM["vocab_size"], LM["hidden_size"], dtype=torch.float32)
        rows_stored = stored["embedding_rows"]
        values = torch.from_numpy(np.frombuffer(base64.b64decode(stored["embedding_values"]), dtype="<f4").copy()).reshape(len(rows_stored), -1)
        emb[rows_stored] = values
        shapes = {k: v.shape for k, v in build_lm(rows).state_dict().items()}
        lm32 = {k: torch.tensor(v, dtype=torch.float32).reshape(shapes[k]) for k, v in stored["lm_state"].items()}
        lm32["model.embed_tokens.weight"] = emb
    lm = build_lm(rows, {k: v for k, v in lm32.items()})
    with torch.no_grad():
        out = lm.generate(torch.tensor([prompt_ids]), do_sample=False, max_new_tokens=24, repetition_penalty=1.1,
                          eos_token_id=tokenizer.token_to_id("<|im_end|>"), pad_token_id=tokenizer.token_to_id("<|endoftext|>"))
        generated = out[0, len(prompt_ids):].tolist()
        ids = torch.tensor([training_ids])
        labels = ids.clone()
        labels[0, :header_len] = -100
        # Precision patch: transformers' loss casts the logits to float32; the same shifted cross-entropy in float64.
        logits = lm(input_ids=ids).logits
        loss = float(torch.nn.functional.cross_entropy(logits[0, :-1], labels[0, 1:], ignore_index=-100))

    torch.manual_seed(17)
    import dac
    dac_model = dac.DAC(**DAC)
    if not args.write:
        dac_model.load_state_dict({k: torch.tensor(v, dtype=torch.float32).reshape(dac_model.state_dict()[k].shape)
                                   for k, v in stored["dac_state"].items()}, strict=True)
    dac_state = {k: v.float().clone() for k, v in dac_model.state_dict().items()}
    dac_model = dac_model.double().eval()
    ap = audio_processor.AudioProcessor.__new__(audio_processor.AudioProcessor)
    ap.device = torch.device("cpu")
    ap.features = audio_processor.Features("cpu")
    codec = dac_interface.DacInterface.__new__(dac_interface.DacInterface)
    codec.device = torch.device("cpu")
    codec.model = dac_model
    codec.sr = 24000
    ap.audio_codec = codec
    t = np.arange(int(2.5 * 24000)) / 24000
    audio = 0.3 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t)) + 0.05 * np.sin(2 * np.pi * 1300 * t)
    codec.load_audio = lambda _: torch.from_numpy(audio).double().reshape(1, 1, -1)
    with torch.no_grad():
        created = ap.create_speaker_from_dict({"audio": {"bytes": b""}, "text": "Hello there, how are you?",
                                               "words": [{"word": w, "start": s, "end": e} for w, s, e in WORDS]})
    torch.set_default_dtype(torch.float32)

    data = {"lm_config": LM, "dac_config": DAC, "prompt_ids": prompt_ids, "generated": generated, "training_ids": training_ids,
            "header_length": header_len, "loss": loss, "audio": b64(audio, "<f8"), "words": WORDS, "speaker": created}
    if args.write:
        emb = lm32.pop("model.embed_tokens.weight")
        data["embedding_rows"] = rows
        data["embedding_values"] = b64(emb[rows].numpy(), "<f4")
        data["lm_state"] = {k: v.reshape(-1).tolist() for k, v in lm32.items()}
        data["dac_state"] = {k: v.reshape(-1).tolist() for k, v in dac_state.items()}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, "loss", loss, "generated", generated[:8])
        return
    check = _fixture.Comparison("OuteTTS model")
    check.exact("generated", stored["generated"], generated)
    check.floats("loss", [stored["loss"]], [loss], 1e-12, float)
    check.exact("speaker", stored["speaker"], json.loads(json.dumps(created)))
    check.report(exact=False)


if __name__ == "__main__":
    main()
