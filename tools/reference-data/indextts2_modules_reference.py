"""IndexTTS2's text-to-semantic model, semantic-to-mel module and BigVGAN-v2 vocoder, from the release's code, on tiny sizes.

Builds the release's UnifiedVoice (indextts/gpt/model_v2.py: the conformer and perceiver conditioners, the emotion
conditioner, GPT-2), its flow-matching estimator and length regulator (indextts/s2mel/modules: CFM, DiT,
InterpolateRegulator) with the GPT latent MLP, and BigVGAN-v2 (indextts/s2mel/modules/bigvgan) with SnakeBeta, and
records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/indextts2_modules_reference.json, in float64 on float32 weights:
- the speaker latents and the emotion vector of w2v-BERT-like features;
- the semantic latents of a text and codes under the release's speed tokens (UnifiedVoice.forward) and under the paper's
  duration token (p = W_sem[T], and zeros), and their semantic logits;
- the codes of the release's generate (beam search over 3 beams, repetition penalty 10, no sampling);
- the length regulator's output, the DiT's velocity, and the Euler solution with classifier-free guidance;
- BigVGAN-v2's waveform.
Default: verify; --write: regenerate. Pass --indextts (the index-tts checkout). Needs torch 2.8, transformers 4.52.1,
einops, omegaconf.
"""
import argparse
import os
import sys

import torch

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/indextts2_modules_reference.json"

D, HEADS, LAYERS, MAX_TEXT, MAX_MEL, TEXT_TOKENS, MEL_CODES = 32, 2, 2, 20, 40, 20, 24
START_MEL, STOP_MEL = 22, 23
COND = dict(output_size=16, linear_units=32, attention_heads=2, num_blocks=2, input_layer="conv2d2", perceiver_mult=2)
EMO = dict(output_size=16, linear_units=24, attention_heads=2, num_blocks=1, input_layer="conv2d2", perceiver_mult=2)
FEATURES = 24   # the release hard-codes w2v-BERT's 1024 (conformer inputs, emotion perceiver); narrowed below
FRAMES, EMO_FRAMES = 13, 9
TEXT = [3, 7, 11, 4, 9]
CODES = [5, 17, 2, 2, 9, 13]
MELS, DIT_H, DIT_HEADS, DIT_DEPTH, STYLE, SEM, REG = 5, 16, 2, 5, 6, 12, 8
WN_LAYERS, GPT_A, GPT_B = 3, 10, 7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indextts", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.indextts))
    # model_v2 imports the v2.5 language table (through openai-whisper) only for its CAM++ speaker mode; stub it.
    import types
    sys.modules["indextts.utils.tokenizer"] = types.SimpleNamespace(LANGUAGE_DICT={})
    # The length regulator imports DAC's VectorQuantize (through descript-audiotools) for vector_quantize, which the
    # release turns off.
    for name in ("indextts.s2mel.dac", "indextts.s2mel.dac.nn"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["indextts.s2mel.dac.nn.quantize"] = types.SimpleNamespace(VectorQuantize=None)
    import indextts.s2mel.modules.gpt_fast.model as gpt_fast
    import indextts.s2mel.modules.diffusion_transformer as dit_module
    import indextts.gpt.model_v2 as model_v2
    from indextts.gpt.model_v2 import UnifiedVoice
    # UnifiedVoice builds its conditioners on 1024 w2v-BERT features and a 1024-wide emotion perceiver; narrow both to
    # FEATURES so the fixture stays small (the port takes these widths as parameters).
    conformer, perceiver = model_v2.ConformerEncoder, model_v2.PerceiverResampler
    model_v2.ConformerEncoder = lambda input_size, **kw: conformer(input_size=FEATURES, **kw)
    model_v2.PerceiverResampler = lambda dim, **kw: perceiver(FEATURES if dim == 1024 else dim, **kw)
    from indextts.s2mel.modules.flow_matching import CFM
    from indextts.s2mel.modules.length_regulator import InterpolateRegulator
    from indextts.s2mel.modules.bigvgan.bigvgan import BigVGAN
    from indextts.s2mel.modules.bigvgan.env import AttrDict
    from omegaconf import OmegaConf

    # Precision patches: the release computes these in float32 whatever the model's dtype; the same in float64.
    def precompute_freqs_cis(seq_len, n_elem, base=10000, dtype=None):
        freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float64)[: (n_elem // 2)] / n_elem))
        t = torch.arange(seq_len, dtype=torch.float64)
        freqs = torch.outer(t, freqs)
        return torch.stack([freqs.cos(), freqs.sin()], dim=-1)
    gpt_fast.precompute_freqs_cis = precompute_freqs_cis

    def apply_rotary_emb(x, freqs_cis):
        xs = x.reshape(*x.shape[:-1], -1, 2)
        f = freqs_cis.view(1, xs.size(1), 1, xs.size(3), 2)
        out = torch.stack([xs[..., 0] * f[..., 0] - xs[..., 1] * f[..., 1], xs[..., 1] * f[..., 0] + xs[..., 0] * f[..., 1]], -1)
        return out.flatten(3)
    gpt_fast.apply_rotary_emb = apply_rotary_emb
    gpt_fast.RMSNorm.forward = lambda self, x: x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps) * self.weight

    def timestep_embedding(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0, dtype=torch.float64)) * torch.arange(half, dtype=torch.float64) / half)
        a = self.scale * t[:, None] * freqs[None]
        return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
    dit_module.TimestepEmbedder.timestep_embedding = timestep_embedding

    stored = None if args.write else _fixture.load(FIXTURE)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2506)

    gpt = UnifiedVoice(layers=LAYERS, model_dim=D, heads=HEADS, max_text_tokens=MAX_TEXT, max_mel_tokens=MAX_MEL,
                       number_text_tokens=TEXT_TOKENS, number_mel_codes=MEL_CODES, start_mel_token=START_MEL, stop_mel_token=STOP_MEL,
                       condition_type="conformer_perceiver", condition_module=COND, emo_condition_module=EMO)
    s2mel_args = OmegaConf.create({
        "reg_loss_type": "l1", "style_encoder": {"dim": STYLE},
        "DiT": dict(hidden_dim=DIT_H, num_heads=DIT_HEADS, depth=DIT_DEPTH, class_dropout_prob=0.1, block_size=64, in_channels=MELS,
                    style_condition=True, final_layer_type="wavenet", target="mel", content_dim=REG, content_codebook_size=16,
                    content_type="discrete", f0_condition=False, n_f0_bins=16, content_codebooks=1, is_causal=False,
                    long_skip_connection=True, zero_prompt_speech_token=False, time_as_token=False, style_as_token=False,
                    uvit_skip_connection=True, add_resblock_in_transformer=False),
        "wavenet": dict(hidden_dim=DIT_H, num_layers=WN_LAYERS, kernel_size=5, dilation_rate=1, p_dropout=0.2, style_condition=True),
        "dit_type": "DiT",
    })
    vocoder_h = AttrDict(dict(resblock="1", upsample_rates=[2, 2], upsample_kernel_sizes=[4, 4], upsample_initial_channel=8,
                              resblock_kernel_sizes=[3], resblock_dilation_sizes=[[1, 3]], use_tanh_at_final=False,
                              use_bias_at_final=False, activation="snakebeta", snake_logscale=True, num_mels=MELS))
    gpt.emovec_layer = torch.nn.Linear(FEATURES, D)
    m = torch.nn.ModuleDict({
        "gpt": gpt,
        "regulator": InterpolateRegulator(channels=REG, sampling_ratios=[1, 1], is_discrete=False, in_channels=SEM, codebook_size=16),
        "cfm": CFM(s2mel_args),
        "gpt_layer": torch.nn.Sequential(torch.nn.Linear(D, GPT_A), torch.nn.Linear(GPT_A, GPT_B), torch.nn.Linear(GPT_B, SEM)),
        "vocoder": BigVGAN(vocoder_h),
    })
    m.eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if stored is not None:
                p.copy_(torch.tensor(stored["state"][name], dtype=torch.float32).reshape(p.shape))
            else:
                scale = 0.5 if p.dim() == 1 else 1.0
                if "speed_emb" in name:
                    p.copy_((0.2 * torch.randn_like(p)).float())
                elif name.endswith("pos_bias_u") or name.endswith("pos_bias_v") or "norm" in name or name.endswith(".gamma"):
                    p.copy_((1.0 + 0.2 * torch.randn_like(p) if ("norm" in name and name.endswith("weight")) or name.endswith(".gamma")
                             else 0.2 * torch.randn_like(p)).float())
                else:
                    p.copy_((p + 0.05 * scale * torch.randn_like(p)).float())
    gpt.post_init_gpt2_config(kv_cache=True)

    def tensor(key, shape, std=1.0):
        if stored is not None:
            return torch.tensor(stored[key], dtype=torch.float32).reshape(shape).to(torch.float64)
        return (std * torch.randn(shape)).float().to(torch.float64)

    shapes = {"features": (FRAMES, FEATURES), "emo_features": (EMO_FRAMES, FEATURES), "content": (6, SEM), "gpt_latents": (6, D),
              "x": (11, MELS), "prompt_mel": (4, MELS), "style": (STYLE,), "noise": (11, MELS), "vocoder_mel": (MELS, 6)}
    inputs = {key: tensor(key, shape) for key, shape in shapes.items()}
    features, emo_features = inputs["features"], inputs["emo_features"]
    out = {}
    with torch.no_grad():
        speaker = gpt.get_conditioning(features.unsqueeze(0).transpose(1, 2), torch.tensor([FRAMES]))[0]   # takes [B, features, T]
        out["speaker"] = speaker
        emovec = gpt.get_emovec(emo_features.unsqueeze(0), torch.tensor([EMO_FRAMES]))
        out["emotion"] = emovec[0]
        text = torch.tensor([TEXT])
        codes = torch.tensor([CODES])
        # The release (speed tokens): UnifiedVoice.forward's latents, given the speaker latents.
        latent = gpt(speaker.unsqueeze(0), text, torch.tensor([len(TEXT)]), codes, torch.tensor([len(CODES)]), None,
                     emo_vec=emovec, use_speed=torch.zeros(1).long())
        out["release_latents"] = latent[0]

        # The paper's layout [c + e, p, BT text ET, BA codes]: full latents (one per semantic input) and logits.
        def paper(p):
            conds = torch.cat([speaker.unsqueeze(0) + emovec.unsqueeze(1), p.view(1, 1, D)], 1)
            ti = torch.nn.functional.pad(torch.nn.functional.pad(text, (1, 0), value=gpt.start_text_token), (0, 1), value=gpt.stop_text_token)
            text_emb = gpt.text_embedding(ti) + gpt.text_pos_embedding(ti)
            mi = torch.nn.functional.pad(codes, (1, 0), value=START_MEL)
            mel_emb = gpt.mel_embedding(mi) + gpt.mel_pos_embedding(mi)
            emb = torch.cat([conds, text_emb, mel_emb], 1)
            h = gpt.gpt(inputs_embeds=emb, return_dict=True).last_hidden_state[:, -mi.shape[1]:]
            h = gpt.final_norm(h)
            return h[0], gpt.mel_head(h)[0]
        lat, logits = paper(gpt.mel_pos_embedding.emb.weight[len(CODES) + 1])
        out["paper_latents"], out["paper_logits"] = lat, logits
        lat0, _ = paper(torch.zeros(D))
        out["paper_free_latents"] = lat0

        # inference_speech computes the speaker latents from the features itself.
        generated, _ = gpt.inference_speech(features.unsqueeze(0), text, None,   # takes [B, T, features], as infer_v2 passes it
                                            cond_lengths=torch.tensor([FRAMES]), emo_vec=emovec, do_sample=False,
                                            num_beams=3, repetition_penalty=10.0, length_penalty=0.0, max_generate_length=10)
        g = generated[0].tolist()
        out_codes = g[:g.index(STOP_MEL)] if STOP_MEL in g else g

        content = inputs["content"]
        frames = 11
        regulated = m["regulator"](content.unsqueeze(0), ylens=torch.tensor([frames]))[0][0]
        out["regulated"] = regulated
        fused = content + m["gpt_layer"](inputs["gpt_latents"])
        out["fused"] = fused

        x = inputs["x"]
        prompt_len = 4
        prompt_mel = inputs["prompt_mel"]
        prompt_x = torch.zeros(frames, MELS)
        prompt_x[:prompt_len] = prompt_mel
        style = inputs["style"]
        est = m["cfm"].estimator
        est.setup_caches(max_batch_size=2, max_seq_length=64)
        velocity = est(x.T.unsqueeze(0), prompt_x.T.unsqueeze(0), torch.tensor([frames]), torch.tensor([0.37]), style.unsqueeze(0),
                       regulated.unsqueeze(0))[0].T
        out["velocity"] = velocity
        noise = inputs["noise"]
        solved = m["cfm"].solve_euler(noise.T.unsqueeze(0).clone(), torch.tensor([frames]), prompt_mel.T.unsqueeze(0), regulated.unsqueeze(0),
                                      style.unsqueeze(0), None, torch.linspace(0, 1, 4 + 1), inference_cfg_rate=0.7)[0].T[prompt_len:]
        out["solved"] = solved

        vocoder_mel = inputs["vocoder_mel"]
        out["waveform"] = m["vocoder"](vocoder_mel.unsqueeze(0))[0, 0]

    if args.write:
        data = {"state": {n: _fixture.flat(p) for n, p in m.named_parameters()}, "codes": out_codes}
        data.update({key: _fixture.flat(value) for key, value in inputs.items()})
        data.update({key: [float(v) for v in value.reshape(-1).tolist()] for key, value in out.items()})
        _fixture.save(FIXTURE, data, __file__)
        print(f"wrote {FIXTURE}: codes {out_codes}")
        return
    compare(stored, out, out_codes)


def compare(stored, out, codes):
    check = _fixture.Comparison("IndexTTS2 modules")
    for key, value in out.items():
        check.floats(key, stored[key], [float(v) for v in value.reshape(-1).tolist()], dtype="float64")
    check.exact("codes", stored["codes"], codes)
    check.report()


if __name__ == "__main__":
    main()
