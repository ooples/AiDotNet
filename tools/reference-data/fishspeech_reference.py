"""Reference fixture for Fish Speech's networks, built small, in float64, from a release's own code: the Dual-AR transformer
(slow and fast transformers) and, for 1.5, the Firefly-GAN codec (log mel, ConvNeXt encoder, grouped FSQ, HiFi-GAN head
with ParallelBlocks). Run once per release: --version 1.5 with the v1.5.1 checkout, --version 1.4 with v1.4.3.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/fishspeech{14,15}_reference.json:
- a training sample as the release's dataset packs it (pack_sentences) and the model's token and codebook logits on it,
  with lit_module's two cross-entropies;
- a voice-cloning prompt as generate_long assembles it (the reference's text and codes, the text to speak) and the
  release's generate on it at top-p 1e-9 (the most likely token each step; the release samples), with the codes
  generate_long keeps;
- the sequence log-probabilities of two sequences under the model and under a second model (the DPO reference), with
  the standard DPO loss (beta 0.1) and Fish Speech's own reference-free one (git 8093258: beta 0.2 on the codebook
  log-probabilities);
- (1.5) Firefly-GAN's log mel, codes and decoded audio.

Weights are closed forms of each parameter's name and index (maskgct_reference.py's, the transformers' scaled by 3 so the
tiny model's greedy choices vary; a quantizer's input projection by 8 so its codes do); the DPO reference model's are 0.9
of the model's. The tokenizer is a stub (characters to ids 0-19, the chat and semantic tokens after them), so the C# test
needs no vocabulary. Precision patches: RMSNorm and the rotary embedding keep float64 (the release casts to float32);
the rotary table stays bfloat16, as the release stores it; attention is computed explicitly (torch 2.4.1's CPU kernel
returns NaN for float64 under a boolean mask).

Default: verify; --write: regenerate. Usage: python fishspeech_reference.py --fish <checkout> --version 1.5|1.4.
Needs .venv-fish (torch 2.4.1, transformers 4.45.2, vector_quantize_pytorch 1.14.24, einx 0.2.2, loguru, tiktoken).
"""
import argparse
import math
import os
import re
import sys
import types

import torch
import torch.nn.functional as F

import _fixture

TEXT_IDS = 20
MODEL_SCALE = 3.0      # the transformers' weights, so the tiny model's greedy choices vary (and reach semantic tokens)


def closed_form(name, shape, scale=1.0):
    count = math.prod(shape)
    k = torch.arange(count, dtype=torch.int64)
    i = k.to(torch.float64)
    s = torch.sin(0.37 * i + 0.11 * len(name) + 0.0013 * ((k * k) % 9973).to(torch.float64))
    if name.endswith(("to_weight.bias", "norm.weight")):
        return scale * (1 + 0.2 * s)
    if name.endswith("project_in.weight"):
        return scale * 8 * s / math.sqrt(count / shape[0])
    if name.endswith("original0"):
        return scale * (0.5 + 0.2 * s)
    if len(shape) >= 2:
        return scale * 0.6 * s / math.sqrt(count / shape[0])
    return scale * 0.05 * s


def fill(model, scale=1.0):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.numel():
                p.copy_(closed_form(name, list(p.shape), scale).reshape(p.shape))
    return {name: list(p.shape) for name, p in model.named_parameters() if p.numel()}


class StubTokenizer:
    """Characters to ids 0-19; the chat and semantic tokens after them."""

    def __init__(self, special):
        self.special = special

    def ids(self, text):
        out = []
        for piece in re.split(r"(<\|[^|]*\|>)", text):
            if not piece:
                continue
            if piece in self.special:
                out.append(self.special[piece])
            else:
                out.extend(ord(c) % TEXT_IDS for c in piece)
        return out

    # 1.5's FishTokenizer
    def encode(self, text, *args, return_tensors=None, **kwargs):
        ids = self.ids(text)
        return torch.tensor([ids]) if return_tensors == "pt" else ids

    def get_token_id(self, token):
        return self.special.get(token, 10_000 + hash(token) % 1000)

    # 1.4's AutoTokenizer
    def convert_tokens_to_ids(self, tokens):
        return [self.special[t] for t in tokens] if isinstance(tokens, list) else self.special[tokens]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fish", required=True)
    parser.add_argument("--version", required=True, choices=["1.4", "1.5"])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    v15 = args.version == "1.5"
    fixture = f"tests/AiDotNet.Tests/TextToSpeech/ReferenceData/fishspeech{'15' if v15 else '14'}_reference.json"
    root = os.path.abspath(args.fish)
    sys.path.insert(0, root)
    torch.set_default_dtype(torch.float64)
    import fish_speech.models.text2semantic.llama as llama

    def rms_norm64(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps) * self.weight

    def rotary64(x, freqs_cis):
        xshaped = x.reshape(*x.shape[:-1], -1, 2)
        freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2).to(x.dtype)
        out = torch.stack([xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
                           xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1]], -1)
        return out.flatten(3)
    llama.RMSNorm.forward = rms_norm64
    llama.apply_rotary_emb = rotary64
    # torch 2.4.1's CPU flash kernel returns NaN for float64 attention under a boolean mask (the release runs in bfloat16 or
    # float32, which it handles); the math kernel is the same attention.
    def attention64(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        if is_causal:
            attn_mask = torch.ones(q.size(-2), k.size(-2), dtype=torch.bool).tril()
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else scores + attn_mask
        return torch.softmax(scores, dim=-1) @ v
    F.scaled_dot_product_attention = attention64

    codebook_size, num_codebooks = 16, 3
    if v15:
        special = {"<|im_start|>": 20, "<|im_end|>": 21, "<|voice|>": 22, "<|end_of_text|>": 23}
        semantic_begin = 24
        special.update({f"<|semantic:{i}|>": semantic_begin + i for i in range(codebook_size)})
        vocab = semantic_begin + codebook_size
    else:
        special = {"<|im_start|>": 20, "<|im_end|>": 21, "<|semantic|>": 22}
        vocab = 24
    tokenizer = StubTokenizer(special)
    if v15:
        tokenizer.semantic_begin_id = semantic_begin
        tokenizer.semantic_id_to_token_id = {i: semantic_begin + i for i in range(codebook_size)}
    config = llama.DualARModelArgs(model_type="dual_ar", vocab_size=vocab, n_layer=2, n_head=4, dim=32, intermediate_size=48,
                                   n_local_heads=2, rope_base=1e6, norm_eps=1e-6, max_seq_len=192, dropout=0.0, tie_word_embeddings=False,
                                   codebook_size=codebook_size, num_codebooks=num_codebooks, n_fast_layer=2, use_gradient_checkpointing=False)
    model = llama.DualARTransformer(config, tokenizer)
    shapes = fill(model, MODEL_SCALE)
    model.eval()
    reference = llama.DualARTransformer(config, tokenizer)
    fill(reference, 0.9 * MODEL_SCALE)
    reference.eval()

    codes = [[(3 * j + 5 * b) % 13 for j in range(6)] for b in range(num_codebooks)]
    out, results = {}, {}

    # ---- the training sample, as the release's dataset packs it
    if v15:
        from fish_speech.datasets.semantic import AutoTextSemanticInstructionIterableDataset as Dataset
    else:
        from fish_speech.datasets.semantic import AutoTextSemanticInstructionDataset as Dataset
    holder = types.SimpleNamespace(tokenizer=tokenizer, num_codebooks=num_codebooks, semantic_token_id=special.get("<|semantic|>"))
    semantics = [[types.SimpleNamespace(values=row) for row in codes]]
    tokens, labels = Dataset.pack_sentences(holder, ["Hi there."], semantics)
    results["train_tokens"] = tokens.tolist()
    results["train_labels"] = labels.tolist()

    def forward(m, tokens):
        inp = tokens.unsqueeze(0)
        return m(inp, key_padding_mask=torch.zeros(1, inp.shape[-1], dtype=torch.bool))

    def losses(result, labels):
        lab = labels.unsqueeze(0)
        base = F.cross_entropy(result.token_logits.view(-1, result.token_logits.size(-1)), lab[:, 0].reshape(-1), ignore_index=-100)
        codebook_labels = lab[:, 1:1 + num_codebooks].mT
        semantic = F.cross_entropy(result.codebook_logits.view(-1, result.codebook_logits.size(-1)), codebook_labels.reshape(-1),
                                   ignore_index=-100)
        return base, semantic

    def sequence_logps(result, labels):
        # Every labelled main token and codebook token's log-probability, summed.
        lab = labels.unsqueeze(0)
        def gathered(logits, target):
            mask = target != -100
            safe = target.clone()
            safe[~mask] = 0
            per = torch.gather(logits.log_softmax(-1), -1, safe.unsqueeze(-1)).squeeze(-1)
            return (per * mask).sum()
        return gathered(result.token_logits, lab[:, 0]) + gathered(result.codebook_logits, lab[:, 1:1 + num_codebooks].mT)

    with torch.no_grad():
        result = forward(model, tokens)
        out["token_logits"] = result.token_logits[0]
        out["codebook_logits"] = result.codebook_logits[0]
        base, semantic = losses(result, labels)
        out["base_loss"] = base.reshape(1)
        out["semantic_loss"] = semantic.reshape(1)

        # ---- DPO: the training sample against a second one of the same length
        rejected_codes = [[(7 * j + 2 * b + 1) % 11 for j in range(6)] for b in range(num_codebooks)]
        rejected_tokens, rejected_labels = Dataset.pack_sentences(holder, ["Hi there."],
                                                                  [[types.SimpleNamespace(values=row) for row in rejected_codes]])
        results["rejected_tokens"] = rejected_tokens.tolist()
        results["rejected_labels"] = rejected_labels.tolist()
        chosen = forward(model, tokens)
        rejected = forward(model, rejected_tokens)
        policy = torch.stack([sequence_logps(chosen, labels), sequence_logps(rejected, rejected_labels)])
        ref = torch.stack([sequence_logps(forward(reference, tokens), labels), sequence_logps(forward(reference, rejected_tokens), rejected_labels)])
        out["policy_logps"] = policy
        out["reference_logps"] = ref
        out["dpo_loss"] = (-F.logsigmoid(0.1 * ((policy[0] - ref[0]) - (policy[1] - ref[1])))).reshape(1)

        # Fish Speech's reference-free DPO (git 8093258): get_batch_logps on the codebook logits, beta 0.2.
        def batch_logps(logits, lab):
            lab = lab.clone()
            mask = lab != -100
            lab[lab == -100] = 0
            per = torch.gather(logits.log_softmax(-1), dim=-1, index=lab.unsqueeze(-1)).squeeze(-1)
            return (per * mask).sum(-1)
        positive = batch_logps(chosen.codebook_logits, labels.unsqueeze(0)[:, 1:1 + num_codebooks].mT)
        negative = batch_logps(rejected.codebook_logits, rejected_labels.unsqueeze(0)[:, 1:1 + num_codebooks].mT)
        out["reference_free_dpo_loss"] = (-F.logsigmoid((positive - negative) * 0.2).mean()).reshape(1)

        # ---- a voice-cloning prompt and generation, as generate_long runs one segment
        prompt_codes = torch.tensor(codes, dtype=torch.int)
        if v15:
            import fish_speech.models.text2semantic.inference as inference
            from fish_speech.conversation import Conversation, Message, TextPart
            system = Conversation(messages=[Message(role="system", parts=[TextPart(text="Speak out the provided text.")], cal_loss=False)]
                                  ).encode_for_inference(tokenizer=tokenizer, num_codebooks=num_codebooks)
            parts = [system, inference.encode_tokens(tokenizer, "Hello.", device="cpu", prompt_tokens=prompt_codes, num_codebooks=num_codebooks),
                     inference.encode_tokens(tokenizer, "Good day.", device="cpu", num_codebooks=num_codebooks)]
            decode = inference.decode_one_token_ar
            extra = {}
        else:
            sys.path.insert(0, root)
            import tools.llama.generate as inference
            parts = [inference.encode_tokens(tokenizer, "Hello.", device="cpu", prompt_tokens=prompt_codes, num_codebooks=num_codebooks),
                     inference.encode_tokens(tokenizer, "Good day.", device="cpu", num_codebooks=num_codebooks)]
            decode = inference.decode_one_token_ar
            extra = {"im_end_id": special["<|im_end|>"]}
        prompt = torch.cat(parts, dim=1)
        results["prompt"] = prompt.tolist()
        model.setup_caches(max_batch_size=1, max_seq_len=config.max_seq_len, dtype=torch.float64)
        y = inference.generate(model=model, prompt=prompt, max_new_tokens=10, decode_one_token=decode, temperature=0.7, top_p=1e-9,
                               repetition_penalty=1.5, **extra)
        results["generated"] = y.tolist()
        n = prompt.shape[1]
        kept = y[1:, n + 1:] if v15 else y[1:, n:-1] - 1
        results["kept_codes"] = kept.tolist()

    if v15:
        from fish_speech.models.vqgan.modules.firefly import ConvNeXtEncoder, FireflyArchitecture, HiFiGANGenerator
        from fish_speech.models.vqgan.modules.fsq import DownsampleFiniteScalarQuantize
        from fish_speech.utils.spectrogram import LogMelSpectrogram
        codec = FireflyArchitecture(
            backbone=ConvNeXtEncoder(input_channels=12, depths=[1, 1, 1, 1], dims=[8, 8, 8, 16], drop_path_rate=0.0, kernel_size=7),
            head=HiFiGANGenerator(hop_length=16, upsample_rates=[2, 2, 2, 2], upsample_kernel_sizes=[4, 4, 4, 4], resblock_kernel_sizes=[3, 7],
                                  resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]], num_mels=16, upsample_initial_channel=16,
                                  pre_conv_kernel_size=13, post_conv_kernel_size=13),
            quantizer=DownsampleFiniteScalarQuantize(input_dim=16, n_groups=2, n_codebooks=1, levels=[8, 5, 5, 5], downsample_factor=[2, 2]),
            spec_transform=LogMelSpectrogram(sample_rate=8000, n_fft=64, win_length=64, hop_length=16, n_mels=12))
        codec_shapes = fill(codec)
        codec.eval()
        shapes.update({"codec." + k: v for k, v in codec_shapes.items()})
        k = torch.arange(640, dtype=torch.float64)
        stored = None if args.write else _fixture.load(fixture)
        audio = (torch.tensor(stored["audio"]) if stored is not None
                 else (0.4 * torch.sin(0.21 * k) + 0.2 * torch.sin(0.057 * k * (1 + k / 900))).float().double())
        with torch.no_grad():
            mel = codec.spec_transform(audio.unsqueeze(0))
            out["mel"] = mel[0]
            indices, feature_lengths = codec.encode(audio.view(1, 1, -1), torch.tensor([640]))
            results["codec_codes"] = indices[0].tolist()
            decoded, _ = codec.decode(indices, feature_lengths)
            out["decoded"] = decoded[0, 0]
        out_audio = audio
    if args.write:
        data = {"shapes": shapes, **results}
        if v15:
            data["audio"] = _fixture.flat(out_audio)
        data.update({key: [float(v) for v in value.reshape(-1).tolist()] for key, value in out.items()})
        _fixture.save(fixture, data, __file__)
        print(f"wrote {fixture}: generated {results['generated'][0][len(results['prompt'][0]):]}" + (f", codes {results['codec_codes']}" if v15 else ""))
        return
    stored = _fixture.load(fixture)
    check = _fixture.Comparison(f"Fish Speech {args.version}")
    check.exact("shapes", stored["shapes"], shapes)
    for key, value in out.items():
        check.floats(key, stored[key], [float(v) for v in value.reshape(-1).tolist()], dtype="float64")
    for key, value in results.items():
        check.exact(key, stored[key], value)
    check.report()


if __name__ == "__main__":
    main()
