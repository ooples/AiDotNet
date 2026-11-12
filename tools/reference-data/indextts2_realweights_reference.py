"""Real-weights reference for IndexTTS2: the released checkpoints run by the release's own classes (float32, as it runs on
CPU) on a prompt (Amphion's MaskGCT prompt.wav, its first three seconds mixed to mono, stored in the fixture), following
infer_v2.py's infer_generator step by step with the random parts fixed:
- the prompt resampled by torchaudio to 22.05 and 16 kHz; the w2v-BERT 2.0 features (get_emb), their semantic codes
  (semantic_codec.quantize), the reference mel, the CAM++ style;
- the style prompt read as librosa reads it (16 kHz, soxr_hq) and the emotion vector (merge_emovec, alpha 1);
- the speaker latents; generation for "HELLO WORLD." (bpe.model's ids, given directly) by beam search with 3 beams,
  repetition penalty 10 and no sampling; the GPT latents of those codes;
- the flow from given noise (25 Euler steps, guidance 0.7) and BigVGAN-v2's waveform;
- QwenEmotion's answer for a sentence, decoded greedily, and its parsed emotion vector.
Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/indextts2_realweights_reference.json.

Usage: python indextts2_realweights_reference.py --indextts <index-tts> --model <IndexTTS-2> --w2vbert <w2v-bert-2.0>
  --codec <MaskGCT/semantic_codec/model.safetensors> --campplus <campplus_cn_common.bin> --bigvgan <bigvgan_v2_22khz_80band_256x>
  --prompt <prompt.wav>, with torch 2.8, torchaudio, transformers 4.52.1, librosa, soundfile, omegaconf, einops, munch.
"""
import argparse
import base64
import json
import os
import re
import sys
import types

import librosa
import numpy as np
import safetensors.torch
import soundfile
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "indextts2_realweights_reference.json")
TEXT_PIECES = None   # filled from bpe.model
EMOTION_TEXT = "I am so happy to see you again today!"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("indextts", "model", "w2vbert", "codec", "campplus", "bigvgan", "prompt"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.indextts))
    sys.modules["indextts.utils.tokenizer"] = types.SimpleNamespace(LANGUAGE_DICT={})
    for name in ("indextts.s2mel.dac", "indextts.s2mel.dac.nn"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["indextts.s2mel.dac.nn.quantize"] = types.SimpleNamespace(VectorQuantize=None)
    from omegaconf import OmegaConf
    from transformers import AutoModelForCausalLM, AutoTokenizer, SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint
    from indextts.codec.maskgct_codec import build_semantic_codec
    from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
    from indextts.s2mel.modules.bigvgan import bigvgan
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
    from indextts.s2mel.modules.audio import mel_spectrogram
    import sentencepiece

    torch.manual_seed(0)
    cfg = OmegaConf.load(os.path.join(args.model, "config.yaml"))
    wav, rate = soundfile.read(args.prompt, always_2d=True)
    mono = librosa.to_mono(wav.T.astype(np.float32))[: rate * 3]
    audio = torch.tensor(mono).unsqueeze(0)

    extractor = SeamlessM4TFeatureExtractor.from_pretrained(args.w2vbert)
    semantic_model = Wav2Vec2BertModel.from_pretrained(args.w2vbert).eval()
    stats = torch.load(os.path.join(args.model, cfg.w2v_stat))
    mean, std = stats["mean"], torch.sqrt(stats["var"])

    def get_emb(audio16k):
        inputs = extractor(audio16k, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            out = semantic_model(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"], output_hidden_states=True)
        return (out.hidden_states[17] - mean) / std

    codec = build_semantic_codec(cfg.semantic_codec)
    safetensors.torch.load_model(codec, args.codec)
    codec.eval()
    gpt = UnifiedVoice(**cfg.gpt)
    load_checkpoint(gpt, os.path.join(args.model, cfg.gpt_checkpoint))
    gpt.eval()
    gpt.post_init_gpt2_config(kv_cache=True)
    s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel, _, _, _ = load_checkpoint2(s2mel, None, os.path.join(args.model, cfg.s2mel_checkpoint), load_only_params=True, ignore_modules=[], is_distributed=False)
    s2mel.eval()
    s2mel.models["cfm"].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(args.campplus, map_location="cpu"))
    campplus.eval()
    vocoder = bigvgan.BigVGAN.from_pretrained(args.bigvgan, use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    vocoder.eval()
    mel_fn = lambda x: mel_spectrogram(x, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=None, center=False)

    out = {"prompt": torch.tensor(mono), "prompt_rate": int(rate)}
    with torch.no_grad():
        audio_22k = torchaudio.transforms.Resample(rate, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(rate, 16000)(audio)
        spk_cond_emb = get_emb(audio_16k)
        _, s_ref = codec.quantize(spk_cond_emb)
        codes_ref, _ = codec.quantize(spk_cond_emb)
        ref_mel = mel_fn(audio_22k.float())
        feat = torchaudio.compliance.kaldi.fbank(audio_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = campplus(feat.unsqueeze(0))
        prompt_condition = s2mel.models["length_regulator"](s_ref, ylens=torch.LongTensor([ref_mel.size(2)]), n_quantizers=3, f0=None)[0]
        out.update(features=spk_cond_emb[0], prompt_codes=codes_ref.reshape(-1), ref_mel=ref_mel[0].T, style=style[0])

        emo_audio = librosa.resample(mono, orig_sr=rate, target_sr=16000, res_type="soxr_hq")
        emo_cond_emb = get_emb(torch.tensor(emo_audio).unsqueeze(0))
        out["emotion_features"] = emo_cond_emb[0]
        lengths = torch.tensor([spk_cond_emb.shape[-1]])
        emovec = gpt.merge_emovec(spk_cond_emb, emo_cond_emb, lengths, torch.tensor([emo_cond_emb.shape[-1]]), alpha=1.0)
        out["emotion"] = emovec[0]
        speaker = gpt.get_conditioning(spk_cond_emb.transpose(1, 2), lengths)
        out["speaker"] = speaker[0]

        sp = sentencepiece.SentencePieceProcessor(model_file=os.path.join(args.model, "bpe.model"))
        text_ids = sp.Encode("HELLO WORLD.")
        out["text"] = text_ids
        text_tokens = torch.tensor(text_ids, dtype=torch.int32).unsqueeze(0)
        codes, _ = gpt.inference_speech(spk_cond_emb, text_tokens, emo_cond_emb, cond_lengths=lengths,
                                        emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]]), emo_vec=emovec, do_sample=False,
                                        num_beams=3, repetition_penalty=10.0, length_penalty=0.0, max_generate_length=200)
        code = codes[0].tolist()
        code = code[: code.index(gpt.stop_mel_token)] if gpt.stop_mel_token in code else code
        out["codes"] = code
        codes = torch.tensor([code])
        latent = gpt(speaker, text_tokens, torch.tensor([text_tokens.shape[-1]]), codes, torch.tensor([codes.shape[-1]]), emo_cond_emb,
                     cond_mel_lengths=lengths, emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]]), emo_vec=emovec,
                     use_speed=torch.zeros(1).long())
        out["latents"] = latent[0]
        latent = s2mel.models["gpt_layer"](latent)
        s_infer = codec.quantizer.vq2emb(codes.unsqueeze(1)).transpose(1, 2) + latent
        target_lengths = (torch.tensor([len(code)]) * 1.72).long()
        cond = s2mel.models["length_regulator"](s_infer, ylens=target_lengths, n_quantizers=3, f0=None)[0]
        cat = torch.cat([prompt_condition, cond], dim=1)
        total = cat.size(1)
        noise = torch.randn(1, 80, total)
        out["noise"] = noise[0].T
        t_span = torch.linspace(0, 1, 25 + 1)
        mel = s2mel.models["cfm"].solve_euler(noise.clone(), torch.LongTensor([total]), ref_mel, cat, style, None, t_span, inference_cfg_rate=0.7)
        mel = mel[:, :, ref_mel.size(-1):]
        out["mel"] = mel[0].T
        out["waveform"] = vocoder(mel.float()).squeeze()

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model, "qwen0.6bemo4-merge"))
        qwen = AutoModelForCausalLM.from_pretrained(os.path.join(args.model, "qwen0.6bemo4-merge"), torch_dtype=torch.float32).eval()
        messages = [{"role": "system", "content": "文本情感分类"}, {"role": "user", "content": EMOTION_TEXT}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer([prompt], return_tensors="pt")
        generated = qwen.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        out["emotion_prompt"] = prompt
        out["emotion_answer"] = tokenizer.decode(generated[0][len(inputs.input_ids[0]):].tolist(), skip_special_tokens=True)
        out["emotion_text"] = EMOTION_TEXT

    # Float arrays as base64 little-endian float32 (the reference's own precision), integers as lists.
    def plain(v):
        if isinstance(v, list) and v and isinstance(v[0], float):
            v = torch.tensor(v)
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                return base64.b64encode(v.detach().to(torch.float32).contiguous().numpy().astype("<f4").tobytes()).decode("ascii")
            return [int(x) for x in v.reshape(-1).tolist()]
        return v
    data = {k: plain(v) for k, v in out.items()}
    data["shapes"] = {k: list(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)}
    data["generator"] = "tools/reference-data/" + os.path.basename(__file__)
    data["torch"] = torch.__version__
    with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False)
    print("codes", len(data["codes"]), data["codes"][:20], "answer", data["emotion_answer"])


if __name__ == "__main__":
    main()
