"""Fish Speech 1.5 with the released weights (fishaudio/fish-speech-1.5), from the release's own code (v1.5.1), in float64.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/fishspeech15_realweights_reference.json:
- Firefly-GAN on a one-second chirp at 44.1 kHz: the log mel, the codes and the audio decode_vq_tokens decodes from them;
- a voice-cloning prompt as generate_long assembles it with the real tokenizer (the system prompt, the chirp's codes
  under a transcript, the text to speak) and the Dual-AR transformer's token logits at the last position and codebook
  logits at four positions inside the reference's codes;
- the release's generate on that prompt at top-p 1e-9 (the most likely token each step) for four frames.

The precision patches are fishspeech_reference.py's: RMSNorm and the rotary embedding keep float64 (the rotary table
stays the release's bfloat16) and attention is computed explicitly.

Usage: python fishspeech_realweights_reference.py --fish <v1.5.1 checkout> --weights <fish-speech-1.5 directory>
Needs .venv-fish.
"""
import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/fishspeech15_realweights_reference.json"
TRANSCRIPT = "Hello there, this is a test."
TEXT = "Fish speech speaks."
POSITIONS = 4


def patch(llama):
    def rms_norm64(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps) * self.weight

    def rotary64(x, freqs_cis):
        xshaped = x.reshape(*x.shape[:-1], -1, 2)
        freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2).to(x.dtype)
        out = torch.stack([xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
                           xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1]], -1)
        return out.flatten(3)

    def attention64(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        if k.size(1) != q.size(1):
            k = k.repeat_interleave(q.size(1) // k.size(1), dim=1)
            v = v.repeat_interleave(q.size(1) // v.size(1), dim=1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        if is_causal:
            attn_mask = torch.ones(q.size(-2), k.size(-2), dtype=torch.bool).tril()
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else scores + attn_mask
        return torch.softmax(scores, dim=-1) @ v

    llama.RMSNorm.forward = rms_norm64
    llama.apply_rotary_emb = rotary64
    F.scaled_dot_product_attention = attention64


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fish", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.fish))
    torch.set_default_dtype(torch.float64)
    import fish_speech.models.text2semantic.llama as llama
    import fish_speech.models.text2semantic.inference as inference
    from fish_speech.conversation import Conversation, Message, TextPart
    from fish_speech.models.vqgan.modules.firefly import ConvNeXtEncoder, FireflyArchitecture, HiFiGANGenerator
    from fish_speech.models.vqgan.modules.fsq import DownsampleFiniteScalarQuantize
    from fish_speech.utils.spectrogram import LogMelSpectrogram
    patch(llama)

    # ---- Firefly-GAN as configs/firefly_gan_vq.yaml builds it, loaded as vqgan/inference.py's load_model does
    codec = FireflyArchitecture(
        backbone=ConvNeXtEncoder(input_channels=160, depths=[3, 3, 9, 3], dims=[128, 256, 384, 512], drop_path_rate=0.2, kernel_size=7),
        head=HiFiGANGenerator(hop_length=512, upsample_rates=[8, 8, 2, 2, 2], upsample_kernel_sizes=[16, 16, 4, 4, 4],
                              resblock_kernel_sizes=[3, 7, 11], resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], num_mels=512,
                              upsample_initial_channel=512, pre_conv_kernel_size=13, post_conv_kernel_size=13),
        quantizer=DownsampleFiniteScalarQuantize(input_dim=512, n_groups=8, n_codebooks=1, levels=[8, 5, 5, 5], downsample_factor=[2, 2]),
        spec_transform=LogMelSpectrogram(sample_rate=44100, n_mels=160, n_fft=2048, hop_length=512, win_length=2048))
    state = torch.load(os.path.join(args.weights, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"), map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    if any("generator" in k for k in state):
        state = {k.replace("generator.", ""): v for k, v in state.items() if "generator." in k}
    result = codec.load_state_dict(state, strict=False)
    assert not result.missing_keys, result.missing_keys
    codec = codec.double().eval()

    k = torch.arange(44100, dtype=torch.float64)
    audio = (0.3 * torch.sin(2 * math.pi * (120 + 300 * k / 44100) * k / 44100)).float().double()
    out, results = {}, {}
    with torch.no_grad():
        out["mel"] = codec.spec_transform(audio.unsqueeze(0))[0]
        indices, lengths = codec.encode(audio.view(1, 1, -1), torch.tensor([audio.numel()]))
        codes = indices[0]
        results["codec_codes"] = codes.tolist()
        # Decoded as inference decodes generated codes (decode_vq_tokens: every frame); encode's feature lengths floor
        # the mel frames and would mask the quantizer's last, padded frame.
        decoded, _ = codec.decode(indices, torch.tensor([indices.shape[2]]))
        out["decoded"] = decoded[0, 0]

    # ---- the Dual-AR transformer
    model = llama.DualARTransformer.from_pretrained(args.weights, load_weights=True).double().eval()
    tokenizer = model.tokenizer
    n = model.config.num_codebooks
    system = Conversation(messages=[Message(role="system", parts=[TextPart(text="Speak out the provided text.")], cal_loss=False)]
                          ).encode_for_inference(tokenizer=tokenizer, num_codebooks=n)
    prompt = torch.cat([system,
                        inference.encode_tokens(tokenizer, TRANSCRIPT, device="cpu", prompt_tokens=codes.int(), num_codebooks=n),
                        inference.encode_tokens(tokenizer, TEXT, device="cpu", num_codebooks=n)], dim=1)
    results["prompt"] = prompt.tolist()
    reference_start = system.shape[1] + inference.encode_tokens(tokenizer, TRANSCRIPT, device="cpu", num_codebooks=n).shape[1]
    positions = [reference_start + 1 + i for i in range(POSITIONS)]
    results["positions"] = positions
    with torch.no_grad():
        forward = model(prompt.unsqueeze(0), key_padding_mask=torch.zeros(1, prompt.shape[1], dtype=torch.bool))
        out["token_logits"] = forward.token_logits[0, -1]
        out["codebook_logits"] = forward.codebook_logits[0, positions]
        model.setup_caches(max_batch_size=1, max_seq_len=prompt.shape[1] + 8, dtype=torch.float64)
        y = inference.generate(model=model, prompt=prompt, max_new_tokens=4, decode_one_token=inference.decode_one_token_ar,
                               temperature=torch.tensor(0.7), top_p=torch.tensor(1e-9), repetition_penalty=torch.tensor(1.2))
        results["generated"] = y[:, prompt.shape[1]:].tolist()

    if args.write:
        data = dict(results)
        data["audio"] = _fixture.flat(audio)
        data.update({key: [float(v) for v in value.reshape(-1).tolist()] for key, value in out.items()})
        _fixture.save(FIXTURE, data, __file__)
        print(f"wrote {FIXTURE}: generated {results['generated']}")
        return
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("Fish Speech 1.5, released weights")
    for key, value in out.items():
        check.floats(key, stored[key], [float(v) for v in value.reshape(-1).tolist()], dtype="float64")
    for key, value in results.items():
        check.exact(key, stored[key], value)
    check.report()


if __name__ == "__main__":
    main()
