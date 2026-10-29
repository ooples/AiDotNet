"""IndexTTS 1.0's released checkpoints run by the release's code, in float64, on a short prompt.

Loads IndexTeam/Index-TTS (gpt.pth, bigvgan_generator.pth, dvae.pth, bpe.model) into the release's UnifiedVoice, BigVGAN
and DVAE and records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/indextts_realweights_reference.json:
- the first two seconds of a prompt at 24 kHz, and the text's tokens and ids (TextTokenizer with TextNormalizer);
- the DVAE's codes of the prompt's log mel;
- slices of the speaker latents, of the latents of the generated codes and of BigVGAN's waveform (with its length), and
  the ECAPA-TDNN speaker embedding;
- the codes of the release's generate without sampling (3 beams, repetition penalty 10, at most 40 codes).
Default: verify; --write: regenerate. Usage: python indextts_realweights_reference.py --indextts <index-tts>
--model <Index-TTS dir> --prompt <wav>. Needs torch 2.8, torchaudio, transformers 4.52.1, WeTextProcessing, sentencepiece.
"""
import argparse
import os
import sys

import torch
import torchaudio

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/indextts_realweights_reference.json"
TEXT = "Hello world, this is IndexTTS speaking."
SECONDS = 2
SLICE = 256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indextts", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.indextts))
    from omegaconf import OmegaConf
    from indextts.gpt.model import UnifiedVoice
    from indextts.BigVGAN.models import BigVGAN
    from indextts.vqvae.xtts_dvae import DiscreteVAE
    from indextts.utils.feature_extractors import MelSpectrogramFeatures
    from indextts.utils.front import TextNormalizer, TextTokenizer

    cfg = OmegaConf.load(os.path.join(args.model, "config.yaml"))
    torch.set_default_dtype(torch.float64)
    gpt = UnifiedVoice(**cfg.gpt)
    state = torch.load(os.path.join(args.model, "gpt.pth"), map_location="cpu")
    gpt.load_state_dict(state.get("model", state), strict=False)
    gpt = gpt.double().eval()
    gpt.post_init_gpt2_config(kv_cache=True)
    bigvgan = BigVGAN(cfg.bigvgan)
    bigvgan.load_state_dict(torch.load(os.path.join(args.model, "bigvgan_generator.pth"), map_location="cpu")["generator"])
    bigvgan = bigvgan.double().eval()
    bigvgan.remove_weight_norm()
    dvae = DiscreteVAE(**cfg.vqvae)
    dvae.load_state_dict(torch.load(os.path.join(args.model, "dvae.pth"), map_location="cpu")["model"])
    dvae = dvae.double().eval()

    normalizer = TextNormalizer()
    normalizer.load()
    tokenizer = TextTokenizer(os.path.join(args.model, "bpe.model"), normalizer)
    tokens = tokenizer.tokenize(TEXT)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    audio, rate = torchaudio.load(args.prompt)
    audio = torch.mean(audio, dim=0, keepdim=True)
    audio = torchaudio.transforms.Resample(rate, 24000)(audio)[:, : SECONDS * 24000].float().to(torch.float64)
    stored = None if args.write else _fixture.load(FIXTURE)
    if stored is not None:
        audio = torch.tensor(stored["prompt"], dtype=torch.float64).unsqueeze(0)
    with torch.no_grad():
        mel = MelSpectrogramFeatures()(audio)                                    # [1, 100, F]
        lengths = torch.tensor([mel.shape[-1]])
        dvae_codes = dvae.get_codebook_indices(mel)[0].tolist()
        speaker = gpt.get_conditioning(mel, lengths)[0]
        text = torch.tensor([ids])
        g = gpt.inference_speech(mel, text, cond_mel_lengths=lengths, do_sample=False, num_beams=3, repetition_penalty=10.0,
                                 length_penalty=0.0, max_generate_length=40)[0].tolist()
        codes = g[:g.index(gpt.stop_mel_token)] if gpt.stop_mel_token in g else g
        c = torch.tensor([codes])
        latent = gpt(mel, text, torch.tensor([len(ids)]), c, torch.tensor([len(codes) * 1024]), cond_mel_lengths=lengths,
                     return_latent=True, clip_inputs=False)
        wave, _ = bigvgan(latent, mel.transpose(1, 2))
        embedding = bigvgan.speaker_encoder(mel.transpose(1, 2)).reshape(-1)
    out = {
        "speaker": speaker.reshape(-1)[:SLICE],
        "latents": latent.reshape(-1)[:SLICE],
        "waveform": wave.reshape(-1)[:4096],
        "speaker_embedding": embedding,
    }
    results = {"tokens": tokens, "ids": ids, "dvae_codes": dvae_codes, "codes": codes, "samples": wave.shape[-1]}
    if args.write:
        data = {"prompt": _fixture.flat(audio[0]), **results}
        data.update({k: [float(v) for v in t.tolist()] for k, t in out.items()})
        _fixture.save(FIXTURE, data, __file__)
        print(f"wrote {FIXTURE}: ids {ids}, codes {codes}, dvae {len(dvae_codes)}")
        return
    check = _fixture.Comparison("IndexTTS real weights")
    for k, t in out.items():
        check.floats(k, stored[k], [float(v) for v in t.tolist()], dtype="float64")
    for k, v in results.items():
        check.exact(k, stored[k], v)
    check.report()


if __name__ == "__main__":
    main()
