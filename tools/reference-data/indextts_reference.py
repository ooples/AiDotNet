"""IndexTTS (v1)'s text-to-mel-code model, BigVGAN decoder and DVAE, from the release's code, on tiny sizes.

Builds the release's UnifiedVoice (indextts/gpt/model.py: the conformer and perceiver conditioner on a 100-band log mel,
GPT-2, the text and mel heads), its BigVGAN (indextts/BigVGAN/models.py: GPT latents in, the ECAPA-TDNN speaker
embedding of the prompt's mel added before and after every upsampling) and its DVAE (indextts/vqvae/xtts_dvae.py), and
records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/indextts_reference.json, in float64 on float32 weights:
- the 100-band log mel (MelSpectrogramFeatures) of a waveform, and the DVAE's encoding and codes of it;
- the speaker latents of that mel;
- UnifiedVoice.forward's text and mel cross-entropies and mel logits on a padded batch item, and its latents
  (return_latent) for the inference call;
- the codes of the release's generate (beam search over 3 beams, repetition penalty 10, no sampling), also for a text
  holding the start and stop ids, which prepare_gpt_inputs drops;
- BigVGAN's waveform of the latents and the prompt's mel;
- remove_long_silence (infer.py) and fade_out_tail (utils/common.py) on fixed inputs.
Default: verify; --write: regenerate. Pass --indextts (the index-tts checkout). Needs torch 2.8, torchaudio,
transformers 4.52.1, einops.
"""
import argparse
import ast
import os
import sys
import types

import torch

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/indextts_reference.json"

D, HEADS, LAYERS, MAX_TEXT, MAX_MEL, TEXT_TOKENS, MEL_CODES = 32, 2, 2, 20, 40, 20, 24
START_MEL, STOP_MEL = 22, 23
COND = dict(output_size=16, linear_units=32, attention_heads=2, num_blocks=2, input_layer="conv2d2", perceiver_mult=2)
SAMPLES = 4000
TEXT = [3, 7, 11, 4, 9]
TEXT_WITH_SPECIALS = [12, 0, 2, 1, 19, 6]
CODES = [5, 17, 2, 2, 9, 13, 8, 6]
WAV_LENGTH = 5 * 1024          # the padded item: codes from index ceil(5) + 1 = 6 on become the stop token
SPEAKER, ECAPA_CHANNELS, ECAPA_ATTENTION, ECAPA_SE = 6, [16, 16, 16, 16, 48], 8, 8
DVAE = dict(positional_dims=1, num_tokens=22, codebook_dim=6, num_layers=2, num_resnet_blocks=1, hidden_dim=8, channels=100,
            stride=2, kernel_size=3, use_transposed_convs=False, smooth_l1_loss=True)
SILENCE_CODES = [4] + [52] * 33 + [7, 52, 52, 9]
FADE_RATE, FADE_SAMPLES = 1000, 30


def release_function(path, class_name, function_name, namespace):
    """The source of one method of the release, executed on its own (importing infer.py pulls in the whole front end)."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == function_name:
                    exec(compile(ast.Module(body=[item], type_ignores=[]), path, "exec"), namespace)
                    return namespace[function_name]
    raise LookupError(f"{class_name}.{function_name} not found in {path}")


def tensors(m):
    """The parameters and the learned buffers (BatchNorm statistics, the DVAE codebook); the anti-aliasing filters and
    attention masks are fixed by construction."""
    buffers = [(n, b) for n, b in m.named_buffers() if n.endswith("running_mean") or n.endswith("running_var") or ".codebook." in n]
    return list(m.named_parameters()) + buffers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indextts", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = os.path.abspath(args.indextts)
    sys.path.insert(0, root)
    from indextts.gpt.model import UnifiedVoice
    import indextts.BigVGAN.models as bigvgan_models
    from indextts.BigVGAN.ECAPA_TDNN import ECAPA_TDNN
    from indextts.vqvae.xtts_dvae import DiscreteVAE
    from indextts.utils.feature_extractors import MelSpectrogramFeatures
    from indextts.utils.common import fade_out_tail

    # The release's ECAPA-TDNN is 512 channels wide (1536 after aggregation); narrowed so the fixture stays small (the
    # port takes the widths as parameters).
    bigvgan_models.ECAPA_TDNN = lambda input_size, lin_neurons: ECAPA_TDNN(
        input_size, lin_neurons=lin_neurons, channels=ECAPA_CHANNELS, attention_channels=ECAPA_ATTENTION, se_channels=ECAPA_SE)

    class AttrDict(dict):
        __getattr__ = dict.__getitem__

    stored = None if args.write else _fixture.load(FIXTURE)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2502)
    gpt = UnifiedVoice(layers=LAYERS, model_dim=D, heads=HEADS, max_text_tokens=MAX_TEXT, max_mel_tokens=MAX_MEL,
                       number_text_tokens=TEXT_TOKENS, number_mel_codes=MEL_CODES, start_mel_token=START_MEL, stop_mel_token=STOP_MEL,
                       condition_type="conformer_perceiver", condition_module=COND)
    # The release's upsampling (x1024, one mel code per 1024 samples) from 64 channels instead of 1536.
    h = AttrDict(resblock="1", upsample_rates=[4, 4, 4, 4, 2, 2], upsample_kernel_sizes=[8, 8, 4, 4, 4, 4], upsample_initial_channel=64,
                 resblock_kernel_sizes=[3], resblock_dilation_sizes=[[1, 3, 5]], feat_upsample=False, speaker_embedding_dim=SPEAKER,
                 cond_d_vector_in_each_upsampling_layer=True, gpt_dim=D, activation="snakebeta", snake_logscale=True, num_mels=100)
    m = torch.nn.ModuleDict({"gpt": gpt, "bigvgan": bigvgan_models.BigVGAN(h), "dvae": DiscreteVAE(**DVAE)})
    m.eval()
    with torch.no_grad():
        for name, p in tensors(m):
            if stored is not None:
                p.copy_(torch.tensor(stored["state"][name], dtype=torch.float32).reshape(p.shape))
            elif name.endswith("running_var"):
                p.copy_((0.5 + torch.rand_like(p)).float())
            elif name.endswith("running_mean"):
                p.copy_((0.3 * torch.randn_like(p)).float())
            elif name.endswith("pos_bias_u") or name.endswith("pos_bias_v") or "norm" in name or name.endswith(".gamma"):
                p.copy_((1.0 + 0.2 * torch.randn_like(p) if ("norm" in name and name.endswith("weight")) or name.endswith(".gamma")
                         else 0.2 * torch.randn_like(p)).float())
            elif name.endswith(".alpha") or name.endswith(".beta"):
                p.copy_((0.3 * torch.randn_like(p)).float())
            elif "dvae.codebook" in name:
                p.copy_((0.15 * torch.randn_like(p)).float())        # the encoder's scale, so the codes vary
            else:
                scale = 0.5 if p.dim() == 1 else 1.0
                p.copy_((p + 0.05 * scale * torch.randn_like(p)).float())
    gpt.post_init_gpt2_config(kv_cache=True)

    def tensor(key, shape, std=1.0):
        if stored is not None:
            return torch.tensor(stored[key], dtype=torch.float32).reshape(shape).to(torch.float64)
        return (std * torch.randn(shape)).float().to(torch.float64)
    inputs = {"wave": tensor("wave", (SAMPLES,), 0.1), "fade_input": tensor("fade_input", (FADE_SAMPLES,))}
    out = {}
    with torch.no_grad():
        mel = MelSpectrogramFeatures()(inputs["wave"].unsqueeze(0))            # [1, 100, F]
        frames = mel.shape[-1]
        out["mel"] = mel[0].T                                                   # [F, 100]
        dvae_codes = m["dvae"].get_codebook_indices(mel)[0].tolist()
        out["dvae_latents"] = m["dvae"].encoder(mel)[0].T                      # [codes, codebook_dim]
        lengths = torch.tensor([frames])
        speaker = gpt.get_conditioning(mel, lengths)[0]
        out["speaker"] = speaker
        text = torch.tensor([TEXT])
        codes = torch.tensor([CODES])
        loss_text, loss_mel, mel_logits = gpt(mel, text, torch.tensor([len(TEXT)]), codes.clone(), torch.tensor([WAV_LENGTH]),
                                              cond_mel_lengths=lengths)
        out["loss_text"] = loss_text.reshape(1)
        out["loss_mel"] = loss_mel.reshape(1)
        out["mel_logits"] = mel_logits[0].T                                     # [n + 2, codes]
        latent = gpt(mel, text, torch.tensor([len(TEXT)]), codes.clone(), torch.tensor([len(CODES) * 1024]), cond_mel_lengths=lengths,
                     return_latent=True, clip_inputs=False)
        out["latents"] = latent[0]                                              # [n, D]

        def generate(text_ids):
            g = gpt.inference_speech(mel, torch.tensor([text_ids]), cond_mel_lengths=lengths, do_sample=False, num_beams=3,
                                     repetition_penalty=10.0, length_penalty=0.0, max_generate_length=12)[0].tolist()
            return g[:g.index(STOP_MEL)] if STOP_MEL in g else g
        generated = generate(TEXT)
        generated_specials = generate(TEXT_WITH_SPECIALS)
        waveform, _ = m["bigvgan"](latent, mel.transpose(1, 2))
        out["waveform"] = waveform[0, 0]
        speaker_embedding = m["bigvgan"].speaker_encoder(mel.transpose(1, 2))
        out["speaker_embedding"] = speaker_embedding.reshape(-1)

        namespace = {"torch": torch, "pad_sequence": torch.nn.utils.rnn.pad_sequence}
        remove_long_silence = release_function(os.path.join(root, "indextts", "infer.py"), "IndexTTS", "remove_long_silence", namespace)
        shrunk, _ = remove_long_silence(types.SimpleNamespace(stop_mel_token=8193), torch.tensor([SILENCE_CODES]), silent_token=52,
                                        max_consecutive=30)
        out["faded"] = fade_out_tail(inputs["fade_input"].unsqueeze(0), FADE_RATE)[0]
    results = {"dvae_codes": dvae_codes, "codes": generated, "codes_specials": generated_specials,
               "silence_shrunk": shrunk[0].tolist()}
    if args.write:
        state = {n: _fixture.flat(p) for n, p in tensors(m)}
        data = {"state": state, **results}
        data.update({key: _fixture.flat(value) for key, value in inputs.items()})
        data.update({key: [float(v) for v in value.reshape(-1).tolist()] for key, value in out.items()})
        data["frames"] = frames
        _fixture.save(FIXTURE, data, __file__)
        print(f"wrote {FIXTURE}: frames {frames}, codes {generated}, specials {generated_specials}, dvae {dvae_codes}")
        return
    compare(stored, out, results)


def compare(stored, out, results):
    check = _fixture.Comparison("IndexTTS")
    for key, value in out.items():
        check.floats(key, stored[key], [float(v) for v in value.reshape(-1).tolist()], dtype="float64")
    for key, value in results.items():
        check.exact(key, stored[key], value)
    check.report()


if __name__ == "__main__":
    main()
