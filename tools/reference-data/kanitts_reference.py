"""KaniTTS (Kani 1) end to end through the release's own code, on a tiny model.

The pieces: kani-tts 0.0.4's KaniModel.get_input_ids (the prompt) and NemoAudioPlayer.get_waveform (the speech span),
the basic example's StreamingAudioWriter (examples/basic/audio/streaming.py over player.py's decode_audio_chunk, run
synchronously), KaniTTS-Finetune-pipeline's TrainDataPreProcessor (add_codes and create_input_ids), transformers 4.57.1's
Lfm2ForCausalLM with generate() (greedy, the repetition penalty, end-of-speech) and its causal LM loss, and NeMo 2.4.0's
NanoCodec modules for the codec (loaded as in nanocodec_reference.py).

LFM2's tokenizer is under its own license, so a small byte-level BPE tokenizer.json is built here with the tokenizers
library (the same pre-tokenizer, decoder and template as LFM2's) and stored in the fixture; both sides read it. The
token layout follows the release's: the text vocabulary, ten control tokens, then four codebooks of codes (here 3·3 = 9
codes each, from a tiny NanoCodec hopping 6 samples).

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/kanitts_reference.json. Pass --kani (nineninesix-ai/kani-tts
checkout), --package (the kani-tts 0.0.4 source, kani_tts-0.0.4/), --finetune (KaniTTS-Finetune-pipeline checkout) and
--nemo (nemo_toolkit-2.4.0 source). Default: verify; --write: regenerate. Needs the .venv-kani environment.
"""
import argparse
import importlib.util
import json
import os
import sys
import types

import numpy as np
import torch

import _fixture
from llama3_qwen3_reference import patch_float64
from nanocodec_reference import load_nemo

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/kanitts_reference.json"
LEVELS = [3, 3]
CODEBOOK = 9
CODEC = dict(encoder=dict(encoded_dim=8, down_sample_rates=[2, 3], base_channels=4, activation="lrelu", pad_mode="replicate"),
             decoder=dict(input_dim=8, up_sample_rates=[3, 2], base_channels=8, activation="half_snake", output_activation="clamp",
                          pad_mode="zeros", n_groups_equal_to_out_channels=True),
             quantizer=dict(num_groups=4, num_levels_per_group=LEVELS))
TEXT = "Hello there, how are you?"
SPEAKER = "Andrew"
CHUNK, LOOKBACK = 3, 2


def build_tokenizer():
    """A byte-level BPE in LFM2's form: <|pad|>, <|startoftext|>, <|endoftext|>, the byte alphabet, a few merges."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors
    from tokenizers.pre_tokenizers import ByteLevel
    alphabet = ByteLevel.alphabet()
    vocab = {"<|pad|>": 0, "<|startoftext|>": 1, "<|endoftext|>": 2}
    for ch in sorted(alphabet):
        vocab[ch] = len(vocab)
    merges = [("Ġ", "t"), ("h", "e"), ("Ġt", "he"), ("l", "l"), ("e", "ll"), ("H", "ell"), ("Hell", "o"), ("Ġ", "a"), ("r", "e"),
              ("o", "u"), ("Ġ", "y"), ("Ġy", "ou"), ("Ġ", "h"), ("o", "w"), ("Ġh", "ow")]
    for a, b in merges:
        vocab[a + b] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    split = ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(split), behavior="isolated", invert=False),
                                                 pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(["<|pad|>", "<|startoftext|>", "<|endoftext|>"])
    tok.post_processor = processors.Sequence([processors.ByteLevel(trim_offsets=False),
                                              processors.TemplateProcessing(single="<|startoftext|> $A", pair="<|startoftext|> $A $B",
                                                                            special_tokens=[("<|startoftext|>", 1)])])
    return tok


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class Codec:
    """AudioCodecModel's encode and decode (nemo/collections/tts/models/audio_codec.py) over NeMo's modules."""

    def __init__(self, m, stored):
        torch.manual_seed(71)
        self.encoder = m.HiFiGANEncoder(**CODEC["encoder"])
        self.decoder = m.CausalHiFiGANDecoder(**CODEC["decoder"])
        self.quantizer = m.GroupFiniteScalarQuantizer(**CODEC["quantizer"])
        with torch.no_grad():
            for prefix, net in (("audio_encoder", self.encoder), ("audio_decoder", self.decoder)):
                for name, p in net.named_parameters():
                    key = f"{prefix}.{name}"
                    if stored is not None:
                        p.copy_(torch.tensor(stored[key], dtype=torch.float32).reshape(p.shape))
                    elif name.endswith("alpha"):
                        p.copy_(0.5 + torch.rand_like(p))
                    elif name == "post_conv.conv.parametrizations.weight.original0":
                        p.copy_(torch.full_like(p, 6.0))
                    elif name.endswith("original0"):
                        p.copy_(0.5 + 0.5 * torch.rand_like(p))
                    elif name.endswith("bias"):
                        p.copy_(0.05 * torch.randn_like(p))
                    else:
                        p.copy_(torch.randn_like(p))
                    p.copy_(p.float())
        self.encoder.double().eval()
        self.decoder.double().eval()
        self.quantizer.double().eval()

    def state(self):
        s = {f"audio_encoder.{k}": v.float().reshape(-1).tolist() for k, v in self.encoder.named_parameters()}
        s.update({f"audio_decoder.{k}": v.float().reshape(-1).tolist() for k, v in self.decoder.named_parameters()})
        return s

    def encode(self, audio, audio_len):
        spf = 6
        padded_len = spf * torch.ceil(audio_len / spf).int()
        padded = torch.nn.functional.pad(audio, (0, padded_len.max().item() - audio.shape[1]))
        encoded, encoded_len = self.encoder(audio=padded, audio_len=padded_len)
        return self.quantizer.encode(inputs=encoded, input_len=encoded_len).permute(1, 0, 2), encoded_len

    def decode(self, tokens, tokens_len):
        dequantized = self.quantizer.decode(indices=tokens.permute(1, 0, 2), input_len=tokens_len)
        return self.decoder(inputs=dequantized, input_len=tokens_len)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("kani", "package", "finetune", "nemo"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    m = load_nemo(os.path.abspath(args.nemo))

    # The release's modules, with their heavy imports (NeMo's model class, datasets) stubbed; nothing they compute is.
    sys.modules["nemo.collections.tts"] = types.ModuleType("nemo.collections.tts")
    sys.modules["nemo.collections.tts.models"] = types.ModuleType("nemo.collections.tts.models")
    sys.modules["nemo.collections.tts.models"].AudioCodecModel = object
    sys.modules["datasets"] = types.ModuleType("datasets")
    sys.modules["datasets"].load_dataset = sys.modules["datasets"].concatenate_datasets = None
    model_config = types.SimpleNamespace(codec_layers=["nano_layer_1", "nano_layer_2", "nano_layer_3", "nano_layer_4"],
                                         codec=types.SimpleNamespace(num_codebooks=4))
    sys.modules["config_loader"] = types.SimpleNamespace(config_loader=types.SimpleNamespace(get_model_config=lambda: model_config))
    core = load_file("kani_core", os.path.join(os.path.abspath(args.package), "kani_tts/core.py"))
    sys.path.insert(0, os.path.join(os.path.abspath(args.kani), "examples/basic"))
    player_module = load_file("kani_player", os.path.join(os.path.abspath(args.kani), "examples/basic/audio/player.py"))
    streaming = load_file("kani_streaming", os.path.join(os.path.abspath(args.kani), "examples/basic/audio/streaming.py"))
    finetune = load_file("kani_finetune", os.path.join(os.path.abspath(args.finetune), "dataset_processor.py"))

    stored = None if args.write else _fixture.load(FIXTURE)
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    tok = build_tokenizer() if args.write else Tokenizer.from_str(stored["tokenizer"])
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok)
    text_vocab = tok.get_vocab_size()
    audio_start = text_vocab + 10
    vocab_size = audio_start + 4 * CODEBOOK
    layout = dict(tokeniser_length=text_vocab, start_of_text=1, end_of_text=2, start_of_speech=text_vocab + 1, end_of_speech=text_vocab + 2,
                  start_of_human=text_vocab + 3, end_of_human=text_vocab + 4, start_of_ai=text_vocab + 5, end_of_ai=text_vocab + 6,
                  pad_token=text_vocab + 7, audio_tokens_start=audio_start, codebook_size=CODEBOOK)

    torch.set_default_dtype(torch.float64)
    codec = Codec(m, None if args.write else stored["codec_state"])

    # kani-tts 0.0.4: the prompt and the speech span.
    player = core.NemoAudioPlayer.__new__(core.NemoAudioPlayer)
    player.__dict__.update(layout)
    player.device = "cpu"
    player.text_tokenizer_name = None
    player.nemo_codec_model = codec
    kani = core.KaniModel.__new__(core.KaniModel)
    kani.player = player
    kani.tokenizer = hf_tokenizer
    prompt = kani.get_input_ids(TEXT, SPEAKER)[0][0].tolist()
    plain_prompt = kani.get_input_ids(TEXT)[0][0].tolist()

    # The language model and greedy generation.
    import transformers.models.lfm2.modeling_lfm2 as lfm2
    patch_float64(lfm2, "Lfm2")
    from transformers import Lfm2Config, Lfm2ForCausalLM
    config = dict(vocab_size=vocab_size, hidden_size=16, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, layer_types=["conv", "full_attention", "conv", "full_attention"], conv_L_cache=3,
                  conv_bias=False, norm_eps=1e-5, rope_theta=1000000.0, max_position_embeddings=256, block_auto_adjust_ff_dim=True,
                  block_ffn_dim_multiplier=1.0, block_multiple_of=8, tie_word_embeddings=True, pad_token_id=0, bos_token_id=1, eos_token_id=7)
    torch.manual_seed(73)
    lm = Lfm2ForCausalLM(Lfm2Config(**config, attn_implementation="eager"))
    with torch.no_grad():
        for name, p in lm.named_parameters():
            if stored is not None:
                if name in stored["lm_state"]:
                    p.copy_(torch.tensor(stored["lm_state"][name], dtype=torch.float32).reshape(p.shape))
            elif name.endswith("norm.weight"):
                p.copy_((1 + 0.1 * torch.randn_like(p)).float())
            else:
                p.copy_((0.3 * torch.randn_like(p)).float())
    lm = lm.double().eval()
    with torch.no_grad():
        generated = lm.generate(torch.tensor([prompt]), attention_mask=torch.ones(1, len(prompt), dtype=torch.long), do_sample=False,
                                max_new_tokens=20, repetition_penalty=1.1, eos_token_id=layout["end_of_speech"], pad_token_id=0)[0, len(prompt):].tolist()

    # A sequence with a speech span of 7 frames (one repeated), for both extractions; the streaming one sees an invalid
    # token (a text id) in its second window.
    rng = np.random.default_rng(5)
    frames = rng.integers(0, CODEBOOK, size=(7, 4))
    frames[3] = frames[2]
    span = (frames + np.array([CODEBOOK * i for i in range(4)]) + audio_start).reshape(-1).tolist()
    sequence = prompt + [layout["start_of_ai"], layout["start_of_speech"]] + span + [layout["end_of_speech"], layout["end_of_ai"]]
    with torch.no_grad():
        span_audio, _ = player.get_waveform(torch.tensor(sequence))
    streamed_tokens = list(span)
    streamed_tokens[13] = 40
    streamed_sequence = prompt + [layout["start_of_ai"], layout["start_of_speech"]] + streamed_tokens + [layout["end_of_speech"], layout["end_of_ai"]]
    audio_player = player_module.LLMAudioPlayer.__new__(player_module.LLMAudioPlayer)
    audio_player.__dict__.update(layout)
    audio_player.device = "cpu"
    audio_player.nemo_codec_model = codec
    writer = streaming.StreamingAudioWriter(audio_player, None, chunk_size=CHUNK, lookback_frames=LOOKBACK)
    for token in streamed_sequence:
        writer.add_token(token)
    writer.running = False
    with torch.no_grad():
        writer.decoder_worker()
    streamed = np.concatenate(writer.audio_chunks) if writer.audio_chunks else np.zeros(0)

    # The fine-tuning pipeline's training example: codes from a recording, repeats removed, the speaker prefixed.
    torch.manual_seed(79)
    audio = (0.5 * torch.randn(1, 40)).float().double() if args.write else torch.tensor([stored["audio"]])
    with torch.no_grad():
        codes, codes_len = codec.encode(audio, torch.tensor([40]))
    codes = codes[0].tolist()
    encoded_codes = [list(c) for c in codes]
    codes[0][2], codes[1][2], codes[2][2], codes[3][2] = codes[0][1], codes[1][1], codes[2][1], codes[3][1]
    processor = finetune.TrainDataPreProcessor.__new__(finetune.TrainDataPreProcessor)
    processor.text_tokenizer = hf_tokenizer
    processor.speaker_id = SPEAKER
    processor.tokeniser_length = text_vocab
    processor.__dict__.update({k: v for k, v in layout.items() if k != "tokeniser_length"})
    example = {"text": TEXT, **{f"nano_layer_{i + 1}": codes[i] for i in range(4)}}
    example = processor.create_input_ids(processor.add_codes(example))
    training_ids = example["input_ids"]
    with torch.no_grad():
        logits = lm(input_ids=torch.tensor([training_ids])).logits
        # Precision patch: transformers' loss casts the logits to float32; the same shifted cross-entropy in float64.
        loss = float(torch.nn.functional.cross_entropy(logits[0, :-1], torch.tensor(training_ids[1:])))
    torch.set_default_dtype(torch.float32)

    data = {"lm_config": config, "codec_config": CODEC, "text": TEXT, "speaker": SPEAKER, "chunk": CHUNK, "lookback": LOOKBACK,
            "text_vocabulary": text_vocab, "prompt": prompt, "plain_prompt": plain_prompt, "generated": generated,
            "sequence": sequence, "span_audio": np.asarray(span_audio).reshape(-1).tolist(), "streamed_sequence": streamed_sequence,
            "streamed_audio": streamed.reshape(-1).tolist(), "audio": audio[0].tolist(), "encoded_codes": encoded_codes, "codes": codes, "training_ids": training_ids,
            "loss": loss}
    if args.write:
        data["tokenizer"] = tok.to_str()
        data["lm_state"] = {k: v.float().reshape(-1).tolist() for k, v in lm.state_dict().items() if k != "lm_head.weight"}
        data["codec_state"] = codec.state()
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, "prompt", prompt, "generated", generated, "streamed", len(streamed), "span", len(data["span_audio"]),
              "training", len(training_ids), "loss", loss)
        return
    check = _fixture.Comparison("KaniTTS")
    for key in ("prompt", "plain_prompt", "generated", "encoded_codes", "codes", "training_ids"):
        check.exact(key, stored[key], data[key])
    check.floats("span audio", stored["span_audio"], data["span_audio"], 1e-12, float)
    check.floats("streamed audio", stored["streamed_audio"], data["streamed_audio"], 1e-12, float)
    check.floats("loss", [stored["loss"]], [loss], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
