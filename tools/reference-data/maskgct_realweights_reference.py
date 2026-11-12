"""Real-weights reference for MaskGCT: the released checkpoints (amphion/MaskGCT at 265c6ce, facebook/w2v-bert-2.0,
Amphion's wav2vec2bert_stats.pt) run by the release's own classes in float64 on the release's prompt (its first three
seconds, mixed to mono as librosa.load does): the prompt resampled to 16 and 24 kHz (librosa.load's soxr_hq, which
SoxrResamplerReferenceTests covers), its semantic
tokens (extract_features, w2v-BERT's 17th hidden state with the mask, the statistics, RepCodec), its acoustic codes and
the audio decoded from them, and one forward of each generator on those tokens (the text-to-semantic model on phones of
the prompt's transcript with half the target masked at t = 0.5, the full acoustic model on layer 3 the same way) — the
logits of a few rows, which pin the loaded weights.

The float32 casts in transformers' rotary embedding, Amphion's adaptive RMSNorm and the eager softmax are replaced as in
maskgct_reference.py.

Usage: python maskgct_realweights_reference.py MASKGCT_DIR W2VBERT_DIR, with transformers 4.41.2, librosa, soundfile and
Amphion next to this repository. Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/maskgct_realweights_reference.json.
"""
import json
import os
import sys
import types

import librosa
import numpy as np
import safetensors.torch
import soundfile
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AMPHION = os.path.join(HERE, "..", "..", "..", "refsrc", "Amphion")
sys.path.insert(0, AMPHION)
sys.modules["LangSegment"] = types.SimpleNamespace(setfilters=lambda filters: None)
torch.set_default_dtype(torch.float64)

import transformers.models.llama.modeling_llama as llama  # noqa: E402
import models.tts.maskgct.llama_nar as llama_nar  # noqa: E402
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel  # noqa: E402
from models.codec.amphion_codec.codec import CodecDecoder, CodecEncoder  # noqa: E402
from models.codec.kmeans.repcodec_model import RepCodec  # noqa: E402
from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A  # noqa: E402
from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S  # noqa: E402


def rotary64(self, x, position_ids):
    inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float64) / self.dim))
    freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].double()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def adaptive_norm64(self, hidden_states, cond_embedding):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    weight = self.to_weight(cond_embedding)
    if len(weight.shape) == 2:
        weight = weight.unsqueeze(1)
    return weight * hidden_states


llama.LlamaRotaryEmbedding.forward = rotary64
llama_nar.LlamaAdaptiveRMSNorm.forward = adaptive_norm64
OFFSETS = [0, 7, 19]   # rows of the target (past the prompt) whose logits are recorded


def load_config(path):
    # The release's config (Amphion's utils.util.load_config reads the same JSON into attribute-access objects).
    def namespace(value):
        if isinstance(value, dict):
            return types.SimpleNamespace(**{k: namespace(v) for k, v in value.items()})
        return value
    with open(path, encoding="utf-8") as handle:
        return namespace(json.load(handle))


def sdpa(module):
    for layer in module.modules():
        if isinstance(layer, llama.LlamaAttention) and not isinstance(layer, llama.LlamaSdpaAttention):
            layer.__class__ = llama.LlamaSdpaAttention


def load(model, path):
    model.load_state_dict(safetensors.torch.load_file(path))   # strict, like load_model
    return model.double().eval()


def main():
    maskgct, w2vbert = sys.argv[1], sys.argv[2]
    cfg = load_config(os.path.join(AMPHION, "models", "tts", "maskgct", "config", "maskgct.json"))
    wav, rate = soundfile.read(os.path.join(AMPHION, "models", "tts", "maskgct", "wav", "prompt.wav"), always_2d=True)
    mono = librosa.to_mono(wav.T.astype(np.float32))[: rate * 3]
    speech16 = librosa.resample(mono, orig_sr=rate, target_sr=16000, res_type="soxr_hq")
    speech24 = librosa.resample(mono, orig_sr=rate, target_sr=24000, res_type="soxr_hq")

    with torch.no_grad():
        extractor = SeamlessM4TFeatureExtractor.from_pretrained(w2vbert)
        inputs = extractor(speech16.astype(np.float64), sampling_rate=16000, return_tensors="pt")
        semantic_model = Wav2Vec2BertModel.from_pretrained(w2vbert).double().eval()
        hidden = semantic_model(input_features=inputs["input_features"].double(), attention_mask=inputs["attention_mask"],
                                output_hidden_states=True).hidden_states[17]
        stats = torch.load(os.path.join(AMPHION, "models", "tts", "maskgct", "ckpt", "wav2vec2bert_stats.pt"))
        hidden = (hidden - stats["mean"].double()) / torch.sqrt(stats["var"]).double()
        del semantic_model
        semantic_codec = load(RepCodec(cfg=cfg.model.semantic_codec), os.path.join(maskgct, "semantic_codec", "model.safetensors"))
        semantic, _ = semantic_codec.quantize(hidden)
        semantic = semantic[0]

        encoder = load(CodecEncoder(cfg=cfg.model.acoustic_codec.encoder), os.path.join(maskgct, "acoustic_codec", "model.safetensors"))
        decoder = load(CodecDecoder(cfg=cfg.model.acoustic_codec.decoder), os.path.join(maskgct, "acoustic_codec", "model_1.safetensors"))
        _, vq, _, _, _ = decoder.quantizer(encoder(torch.from_numpy(speech24.astype(np.float64)).reshape(1, 1, -1)))
        acoustic = vq.permute(1, 2, 0)[0]                                                    # [T, 12]
        audio = decoder(decoder.vq2emb(vq, n_quantizers=12))[0, 0]

        phones = [int(p) for p in (np.arange(24) * 37 % 300 + 3)]
        t2s = load(MaskGCT_T2S(cfg=cfg.model.t2s_model), os.path.join(maskgct, "t2s_model", "model.safetensors"))
        sdpa(t2s)
        n = semantic.shape[0]
        prompt = n // 2
        rows = [prompt + o for o in OFFSETS]
        x0 = semantic[None]
        mask = torch.zeros(1, n, 1)
        mask[0, prompt::2] = 1
        xt = mask * t2s.mask_emb(torch.LongTensor([0]))[:, None, :] + (1 - mask) * t2s.cond_emb(x0)
        embeds = t2s.diff_estimator(xt, torch.tensor([0.5]), torch.ones(1, n), phone_embedding=t2s.phone_emb(torch.LongTensor(phones)[None]),
                                    phone_mask=torch.ones(1, len(phones)))
        t2s_logits = t2s.to_logit(embeds)[0]
        del t2s

        full = load(MaskGCT_S2A(cfg=cfg.model.s2a_model.s2a_full), os.path.join(maskgct, "s2a_model", "s2a_model_full", "model.safetensors"))
        sdpa(full)
        frames = min(n, acoustic.shape[0])
        layer = 3
        codes = acoustic[:frames]
        is_prompt = torch.zeros(frames)
        is_prompt[:prompt] = 1
        layer_mask = torch.zeros(frames)
        layer_mask[prompt::2] = 1
        mask_token = full.mask_emb(torch.LongTensor([0]))
        xs = torch.zeros(1, frames, full.hidden_size)
        for q, emb in enumerate(full.token_emb):
            e = emb(codes[None, :, q])
            if q < layer:
                xs = xs + e
            elif q == layer:
                m = layer_mask[None, :, None]
                xs = xs + m * mask_token[:, None, :] + (1 - m) * e
            else:
                p = is_prompt[None, :, None]
                xs = xs + e * p + mask_token * (1 - p)
        cond = full.cond_emb(semantic[None, :frames]) + full.layer_emb(torch.LongTensor([layer])).unsqueeze(1)
        s2a_logits = full.to_logits[layer](full.diff_estimator(xs, torch.tensor([0.5]), cond, torch.ones(1, frames)))[0]

    result = {
        "rate": rate, "speech16": speech16.astype(np.float64).tolist(),
        "speech24": speech24.astype(np.float64).tolist(), "semantic": semantic.tolist(), "acoustic": acoustic.tolist(),
        "audio_head": audio[:4800].tolist(), "audio_length": int(audio.shape[0]),
        "phones": phones, "prompt": prompt, "rows": rows, "layer": layer,
        "t2s_logits": [t2s_logits[r].tolist() for r in rows], "s2a_logits": [s2a_logits[r].tolist() for r in rows],
    }
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "maskgct_realweights_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle)
        handle.write("\n")
    print(len(mono), len(speech16), len(speech24), n, acoustic.shape, audio.shape)


if __name__ == "__main__":
    main()
