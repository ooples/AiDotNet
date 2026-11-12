"""Parler-TTS reference: huggingface/parler-tts's ParlerTTSForConditionalGeneration, the open reproduction of Lyth & King
(2024) whose code the paper's gaps follow.

Builds tiny randomly initialized models with the reference's code (pass a checkout of huggingface/parler-tts with
--parler; it is not vendored here), stores their weights in float32, runs them in float64 and records, for each case:
- the delayed training labels the reference's training script builds from a code sequence;
- the teacher-forced logits, the loss and each codebook's loss for those labels;
- greedy generation (do_sample=False): the codes the reference hands its audio decoder.
in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/parler_tts_reference.json. Default: verify; --write: regenerate.

The cases are the released configuration (sinusoidal positions, the transcript prepended, a description encoder of
another width so the projection is used, grouped key/value heads) and the reproduction's variants (rotary positions,
the transcript cross-attended). The description encoder is a tiny T5 v1.1 (gated GELU), the form of Flan-T5; the audio
encoder is transformers' DAC, whose decoder generation only calls, so its decode is replaced by a recorder.

parler_tts's package imports descript's `dac` (for its own DAC wrapper); pass --stubs, a directory holding a `dac`
package with a placeholder `DAC` class, when it is not installed. Run with transformers 4.46 (parler-tts pins it).
The sinusoidal and rotary tables are built in float32 by the reference; the float64 runs rebuild them in float64,
which is what the C# port computes.
"""
import argparse
import base64
import math
import sys

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/parler_tts_reference.json"
CODEBOOK_SIZE = 16
EOS = CODEBOOK_SIZE            # also the padding token
BOS = CODEBOOK_SIZE + 1
CASES = {
    "released": dict(rope_embeddings=False, prompt_cross_attention=False),
    "variants": dict(rope_embeddings=True, prompt_cross_attention=True),
}
DECODER = dict(vocab_size=20, max_position_embeddings=64, num_hidden_layers=2, ffn_dim=24, num_attention_heads=4,
               num_key_value_heads=2, num_cross_attention_key_value_heads=1, hidden_size=16, dropout=0.0,
               num_codebooks=3, pad_token_id=EOS, bos_token_id=BOS, eos_token_id=EOS, rope_theta=10_000.0)
TEXT_ENCODER = dict(vocab_size=48, d_model=12, d_kv=4, d_ff=20, num_layers=2, num_heads=3,
                    feed_forward_proj="gated-gelu", tie_word_embeddings=False, dropout_rate=0.0)
PROMPT_VOCABULARY = 40
DESCRIPTION = [5, 17, 33, 2, 41, 9, 1]
TRANSCRIPT = [7, 22, 13, 30, 1]
FRAMES = 6
MAX_LENGTH = 16
MIN_NEW_TOKENS = 3


def load(parler_root, stubs):
    for path in [stubs, parler_root]:
        if path:
            sys.path.insert(0, path)
    import parler_tts
    from parler_tts import modeling_parler_tts as modeling

    def get_embedding_float64(num_embeddings, embedding_dim):
        half_dim = embedding_dim // 2
        step = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, dtype=torch.float64) * -step)
        angles = torch.arange(num_embeddings, dtype=torch.float64).unsqueeze(1) * frequencies.unsqueeze(0)
        table = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)
        if embedding_dim % 2 == 1:
            table = torch.cat([table, torch.zeros(num_embeddings, 1, dtype=torch.float64)], dim=1)
        return table

    @torch.no_grad()
    def rotary_float64(self, device_type, position_ids):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float64) / self.dim))
        angles = position_ids[0].to(torch.float64)[:, None] * inv_freq[None, :]
        table = torch.cat((angles, angles), dim=-1)[None]
        return table.cos(), table.sin()

    modeling.ParlerTTSSinusoidalPositionalEmbedding.get_embedding = staticmethod(get_embedding_float64)
    modeling.ParlerTTSRotaryEmbedding.forward = rotary_float64
    return parler_tts


def config(parler_tts, case):
    from transformers import DacConfig, T5Config
    decoder = parler_tts.ParlerTTSDecoderConfig(**DECODER, rope_embeddings=CASES[case]["rope_embeddings"])
    audio = DacConfig(encoder_hidden_size=4, downsampling_ratios=[2, 2], decoder_hidden_size=8, n_codebooks=3,
                      codebook_size=CODEBOOK_SIZE, codebook_dim=4, hidden_size=8, sampling_rate=16000)
    return parler_tts.ParlerTTSConfig.from_sub_models_config(
        T5Config(**TEXT_ENCODER), audio, decoder, vocab_size=PROMPT_VOCABULARY,
        prompt_cross_attention=CASES[case]["prompt_cross_attention"], pad_token_id=EOS, decoder_start_token_id=BOS)


def computed(key):
    """The sinusoidal tables (the decoder's, and the cross-attended transcript's) are buffers the port computes."""
    return key.endswith("embed_positions.weights")


def build(parler_tts, case, state=None):
    torch.manual_seed(2294 + list(CASES).index(case))
    model = parler_tts.ParlerTTSForConditionalGeneration(config(parler_tts, case))
    if state is None:
        with torch.no_grad():
            # Random heads rarely pick the end token; favour it a little so generation shows codebooks ending in turn.
            for head in model.decoder.lm_heads:
                head.weight[EOS] *= 3.0
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and all(computed(k) or k.startswith("audio_encoder.") for k in missing), (missing, unexpected)
    return model.eval()


def stored_state(model):
    """The weights the C# port reads: everything but the sinusoidal table (a buffer it computes) and the audio encoder
    (only its decoder's interface is used)."""
    return {k: v.float().clone().contiguous() for k, v in model.state_dict().items()
            if not computed(k) and not k.startswith("audio_encoder.")}


def delayed_labels(parler_tts, codes):
    """run_parler_tts_training.py's postprocess_dataset for one utterance, codes [codebooks, frames]."""
    from parler_tts.modeling_parler_tts import build_delay_pattern_mask
    labels = codes.unsqueeze(0)
    labels = torch.cat([torch.full((1, labels.shape[1], 1), BOS, dtype=torch.long), labels], dim=-1)
    labels, mask = build_delay_pattern_mask(labels, bos_token_id=BOS, pad_token_id=EOS,
                                            max_length=labels.shape[-1] + labels.shape[1], num_codebooks=labels.shape[1])
    labels = torch.where(mask == -1, EOS, mask)
    return labels[:, 1:]                                            # [codebooks, frames + codebooks]


def codes_input():
    g = torch.Generator().manual_seed(11)
    return torch.randint(0, CODEBOOK_SIZE, (DECODER["num_codebooks"], FRAMES), generator=g)


def run(parler_tts, model):
    codes = codes_input()
    labels = delayed_labels(parler_tts, codes)
    description = torch.tensor([DESCRIPTION])
    transcript = torch.tensor([TRANSCRIPT])
    with torch.no_grad():
        out = model(input_ids=description, prompt_input_ids=transcript, labels=labels.T[None], return_dict=True)
    recorded = {}

    def decode(audio_codes=None, **kwargs):
        recorded["codes"] = audio_codes.clone()
        return type("Decoded", (), {"audio_values": torch.zeros(1, 1, 8, dtype=torch.float64)})()

    model.audio_encoder.decode = decode
    with torch.no_grad():
        model.generate(input_ids=description, prompt_input_ids=transcript, do_sample=False, max_length=MAX_LENGTH,
                       min_new_tokens=MIN_NEW_TOKENS, decoder_start_token_id=BOS, bos_token_id=BOS, eos_token_id=EOS,
                       pad_token_id=EOS)
    generated = recorded["codes"].reshape(DECODER["num_codebooks"], -1)
    return {
        "labels": labels.tolist(),
        # The code positions' logits [codebooks, length, vocabulary] (the model also returns the prepended transcript's).
        "logits": [float(v) for v in out.logits.reshape(DECODER["num_codebooks"], -1, DECODER["vocab_size"])[:, -labels.shape[1]:].reshape(-1).tolist()],
        "loss": float(out.loss),
        "per_codebook_losses": [float(v) for v in out.per_codebook_losses],
        "generated": generated.T.tolist(),                                    # [frames, codebooks]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parler", required=True, help="a checkout of huggingface/parler-tts")
    parser.add_argument("--stubs", help="a directory holding a placeholder `dac` package")
    parser.add_argument("--write", action="store_true", help="regenerate the fixture (default: verify it)")
    args = parser.parse_args()
    parler_tts = load(args.parler, args.stubs)
    if args.write:
        data = {"decoder": DECODER, "text_encoder": TEXT_ENCODER, "prompt_vocabulary": PROMPT_VOCABULARY,
                "codebook_size": CODEBOOK_SIZE, "description": DESCRIPTION, "transcript": TRANSCRIPT,
                "codes": codes_input().T.tolist(), "max_length": MAX_LENGTH, "min_new_tokens": MIN_NEW_TOKENS, "cases": {}}
        for case in CASES:
            state = stored_state(build(parler_tts, case))
            model = build(parler_tts, case, state).double()
            data["cases"][case] = {**CASES[case], "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                                   **run(parler_tts, model)}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    check = _fixture.Comparison("Parler-TTS")
    for case in CASES:
        committed = data["cases"][case]
        state = load_safetensors(base64.b64decode(committed["safetensors_base64"]))
        again = run(parler_tts, build(parler_tts, case, state).double())
        check.floats(case + " logits", committed["logits"], again["logits"], 1e-12, float)
        check.floats(case + " losses", [committed["loss"], *committed["per_codebook_losses"]],
                     [again["loss"], *again["per_codebook_losses"]], 1e-12, float)
        check.exact(case + " labels", committed["labels"], again["labels"])
        check.exact(case + " generated", committed["generated"], again["generated"])
    check.report(exact=False)


if __name__ == "__main__":
    main()
