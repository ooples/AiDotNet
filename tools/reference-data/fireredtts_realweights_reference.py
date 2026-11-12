"""FireRedTTS-1 with the released weights (fireredteam/FireRedTTS), from the release's own code (main branch), in float64.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/fireredtts_realweights_reference.json:
- the speaker embedding of a one-second 16 kHz chirp (SpeakerEmbedddingExtractor) and the prompt mel of it
  (MelSpectrogramExtractor's mel_spectrogram on the 16 kHz chirp resampled up to 24 kHz and back with kaiser_best);
- the text's token ids (VoiceBpeTokenizer.encode), the GPT's audio logits after [speaker, BT text ET, BA] and the
  release's generate without sampling (repetition penalty 2) for four tokens;
- the flow's mel (three Euler steps from fixed noise, the prompt mel as timbre) of fixed tokens, and BigVGAN's audio of it.
Precision patches as in fireredtts_reference.py (Fp32BatchNorm, the relative position table, the time embedding's
frequencies and the mel filterbank computed in float64, its float32 values kept).

Usage: python fireredtts_realweights_reference.py --firered <main checkout> --weights <fireredtts-hf directory> [--write]
Needs .venv-firered and the text front end's packages (WeTextProcessing has no Windows build: the token ids are taken from
the committed fireredtts_text_reference-style call made in the firered-ref Docker image, see --ids).
"""
import argparse
import json
import math
import os
import sys

import torch

import _fixture
from fireredtts_reference import patch_source

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/fireredtts_realweights_reference.json"
TEXT = "小红书，是中国大陆的网络购物和社交平台。"
FLOW_TOKENS = [1205, 9876, 233, 16000, 4411, 77, 12034, 8191, 5555, 300]
STEPS = 3
GENERATED = 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firered", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--ids", required=True, help="JSON list: the text's token ids from VoiceBpeTokenizer.encode (Docker)")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.firered))
    torch.set_default_dtype(torch.float64)
    import fireredtts.modules.flow.conformer as conformer
    import fireredtts.modules.codec.speaker as speaker_module
    import fireredtts.modules.flow.mel_spectrogram as flow_mel
    import fireredtts.modules.flow.decoder as flow_decoder
    from fireredtts.modules.gpt.gpt import GPT
    from fireredtts.modules import Token2Wav

    def extend_pe64(self, x):
        if self.pe is not None and self.pe.size(1) >= x.size(1) * 2 - 1:
            self.pe = self.pe.to(dtype=x.dtype, device=x.device)
            return
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float64).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float64) * -(math.log(10000.0) / self.d_model))
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
        self.pe = torch.cat([torch.flip(pe_positive, [0]).unsqueeze(0), pe_negative[1:].unsqueeze(0)], dim=1).to(x.dtype)
    conformer.EspnetRelPositionalEncoding.extend_pe = extend_pe64
    speaker_module.Fp32BatchNorm.forward = lambda self, input: self.bn(input)
    patch_source(speaker_module.TorchMelSpectrogram, "forward", "torch.from_numpy(mel).float()", "torch.from_numpy(mel).double()")
    patch_source(flow_mel, "mel_spectrogram", "torch.from_numpy(mel).float()", "torch.from_numpy(mel).double()")
    patch_source(flow_decoder.SinusoidalPosEmb, "forward", "torch.arange(half_dim, device=device).float()",
                 "torch.arange(half_dim, device=device).double()")

    config = json.load(open(os.path.join(os.path.abspath(args.firered), "configs", "config_24k.json")))
    config["flow"]["input_embedding"]["codebook_path"] = os.path.join(os.path.abspath(args.firered), config["flow"]["input_embedding"]["codebook_path"])
    g = config["gpt"]
    gpt = GPT(start_text_token=g["gpt_start_text_token"], stop_text_token=g["gpt_stop_text_token"], layers=g["gpt_layers"],
              model_dim=g["gpt_n_model_channels"], heads=g["gpt_n_heads"], max_text_tokens=g["gpt_max_text_tokens"],
              max_mel_tokens=g["gpt_max_audio_tokens"], max_prompt_tokens=g["gpt_max_prompt_tokens"], code_stride_len=g["gpt_code_stride_len"],
              number_text_tokens=g["gpt_number_text_tokens"], num_audio_tokens=g["gpt_num_audio_tokens"],
              start_audio_token=g["gpt_start_audio_token"], stop_audio_token=g["gpt_stop_audio_token"])
    gpt.load_state_dict(torch.load(os.path.join(args.weights, "fireredtts_gpt.pt"), map_location="cpu", weights_only=False)["model"], strict=True)
    gpt = gpt.double().eval()
    gpt.init_gpt_for_inference(kv_cache=True)
    token2wav = Token2Wav.init_from_config(config)
    token2wav.load_state_dict(torch.load(os.path.join(args.weights, "fireredtts_token2wav.pt"), map_location="cpu", weights_only=False), strict=True)
    token2wav.generator.remove_weight_norm()
    token2wav = token2wav.double().eval()
    for table in token2wav.flow.input_embedding.codebook:
        table.weight.data = table.weight.data.double()
    ecapa = speaker_module.ECAPA_TDNN(80, 512, channels=[512, 512, 512, 512, 1536], kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
                                      attention_channels=128, res2net_scale=4, se_channels=128, global_context=True, batch_norm=True)
    ecapa.load_state_dict(torch.load(os.path.join(args.weights, "fireredtts_speaker.bin"), map_location="cpu", weights_only=False), strict=True)
    ecapa = ecapa.double().eval()

    k = torch.arange(16000, dtype=torch.float64)
    audio = (0.3 * torch.sin(2 * math.pi * (150 + 250 * k / 16000) * k / 16000)).float().double()
    ids = json.loads(args.ids)
    stored = None if args.write else _fixture.load(FIXTURE)
    noise = ((torch.randn(1, 100, 4 * len(FLOW_TOKENS), generator=torch.Generator().manual_seed(11))).float().double()
             if stored is None else torch.tensor(stored["noise"]).reshape(1, 100, 4 * len(FLOW_TOKENS)))
    out, results = {}, {"ids": ids}
    with torch.no_grad():
        mel80 = speaker_module.TorchMelSpectrogram()(audio.unsqueeze(0))
        speaker = ecapa(mel80)[0]
        out["speaker"] = speaker
        resampled = flow_mel.kaiser_best_resampling_fn(audio.unsqueeze(0), orig_freq=16000, new_freq=24000)
        resampled = flow_mel.kaiser_best_resampling_fn(resampled, orig_freq=24000, new_freq=16000)
        prompt_mel = flow_mel.mel_spectrogram(resampled, 1024, 100, 16000, 160, 1024, 0, None)[0].T
        out["prompt_mel"] = prompt_mel
        cond = gpt.reference_embedding(speaker.view(1, 1, 512))
        text = torch.tensor([ids], dtype=torch.int32)
        gpt_inputs, attention_mask = gpt.compute_embeddings(cond, text)
        logits = gpt.gpt_inference(gpt_inputs, attention_mask=attention_mask, return_dict=True).logits[0, -1]
        out["logits"] = logits
        generated = gpt.generate(cond_latents=cond, text_inputs=text, input_tokens=None, do_sample=False, num_beams=1,
                                 repetition_penalty=2.0, max_new_tokens=GENERATED)
        results["generated"] = generated[0].tolist()
        torch.randn_like = lambda x, **kw: noise.clone()
        mel = token2wav.flow.inference(torch.tensor([FLOW_TOKENS]), torch.tensor([len(FLOW_TOKENS)]), prompt_mel.unsqueeze(0),
                                       torch.tensor([prompt_mel.shape[0]]), n_timesteps=STEPS)
        out["flow_mel"] = mel[0].T
        out["audio"] = token2wav.generator(mel)[0, 0]
    if args.write:
        data = dict(results)
        data["chirp"] = _fixture.flat(audio)
        data["noise"] = _fixture.flat(noise)
        data.update({key: [float(v) for v in value.reshape(-1).tolist()] for key, value in out.items()})
        _fixture.save(FIXTURE, data, __file__)
        print(f"wrote {FIXTURE}: generated {results['generated']}")
        return
    check = _fixture.Comparison("FireRedTTS, released weights")
    for key, value in out.items():
        check.floats(key, stored[key], [float(v) for v in value.reshape(-1).tolist()], dtype="float64")
    for key, value in results.items():
        check.exact(key, stored[key], value)
    check.report()


if __name__ == "__main__":
    main()
