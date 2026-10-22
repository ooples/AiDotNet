"""Reference fixture for CosyVoice (Du et al. 2024): the official FunAudioLLM/CosyVoice modules in float64.

Runs the official TransformerLM (text encoder + language model) on one utterance at small sizes: the loss of eq. 8 and
one SGD step. Needs clones of FunAudioLLM/CosyVoice at COSYVOICE_COMMIT and of its Matcha-TTS submodule at
MATCHA_COMMIT next to this repository (../refsrc/CosyVoice, ../refsrc/Matcha-TTS), plus diffusers==0.25.0 and
conformer==0.3.2 installed without dependencies. Packages the modules import but these classes never use (onnxruntime,
whisper, ...) are stubbed.

Numerics: ESPnet's relative positional table is computed in float32 by the reference; here it is computed in float64,
as the C# model computes it, so the comparison measures the network rather than float32 sines.

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/cosyvoice_reference.json. Run with the reference venv.
"""
import importlib.machinery
import json
import math
import os
import sys
from unittest import mock

import huggingface_hub

huggingface_hub.cached_download = huggingface_hub.hf_hub_download  # diffusers 0.25 imports the removed name
for _name in ["onnxruntime", "gdown", "whisper", "tiktoken", "inflect", "pyworld", "wget", "hyperpyyaml", "tensorrt"]:
    try:
        __import__(_name)
    except Exception:
        _stub = mock.MagicMock()
        _stub.__spec__ = importlib.machinery.ModuleSpec(_name, None)
        sys.modules[_name] = _stub

HERE = os.path.dirname(os.path.abspath(__file__))
REFSRC = os.path.normpath(os.path.join(HERE, "..", "..", "..", "refsrc"))
COSYVOICE_COMMIT = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
MATCHA_COMMIT = "dd9105b34bf2be2230f4aa1e4769fb586a3c824e"
sys.path.insert(0, os.path.join(REFSRC, "Matcha-TTS"))
sys.path.insert(0, os.path.join(REFSRC, "CosyVoice"))

import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

from cosyvoice.transformer import embedding as cv_embedding  # noqa: E402
from cosyvoice.transformer.encoder import ConformerEncoder, TransformerEncoder  # noqa: E402
from cosyvoice.llm.llm import TransformerLM  # noqa: E402
from cosyvoice.utils.common import ras_sampling  # noqa: E402


def _extend_pe_float64(self, x):
    # EspnetRelPositionalEncoding.extend_pe with the table in float64 instead of float32.
    if self.pe is not None and self.pe.size(1) >= x.size(1) * 2 - 1:
        return
    pe_positive = torch.zeros(x.size(1), self.d_model)
    pe_negative = torch.zeros(x.size(1), self.d_model)
    position = torch.arange(0, x.size(1), dtype=torch.float64).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float64) * -(math.log(10000.0) / self.d_model))
    pe_positive[:, 0::2] = torch.sin(position * div_term)
    pe_positive[:, 1::2] = torch.cos(position * div_term)
    pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
    pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
    pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
    pe_negative = pe_negative[1:].unsqueeze(0)
    self.pe = torch.cat([pe_positive, pe_negative], dim=1)


cv_embedding.EspnetRelPositionalEncoding.extend_pe = _extend_pe_float64


def flat(module):
    return {k: v.detach().flatten().tolist() for k, v in module.state_dict().items()}


def randomize(module, gen, scale):
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * scale)


LM = dict(text_vocab=20, speech_tokens=12, text_in=8, text_out=16, width=16, heads=2, units=24, blocks=2, spk=6)


def language_model():
    c = LM
    text_encoder = ConformerEncoder(input_size=c["text_in"], output_size=c["text_out"], attention_heads=c["heads"],
                                    linear_units=c["units"], num_blocks=c["blocks"], dropout_rate=0.0,
                                    positional_dropout_rate=0.0, attention_dropout_rate=0.0, normalize_before=True,
                                    input_layer="linear", pos_enc_layer_type="rel_pos_espnet",
                                    selfattention_layer_type="rel_selfattn", use_cnn_module=False,
                                    macaron_style=False, static_chunk_size=1)
    llm = TransformerEncoder(input_size=c["width"], output_size=c["width"], attention_heads=c["heads"],
                             linear_units=c["units"], num_blocks=c["blocks"], dropout_rate=0.0,
                             positional_dropout_rate=0.0, attention_dropout_rate=0.0, input_layer="linear_legacy",
                             pos_enc_layer_type="rel_pos_espnet", selfattention_layer_type="rel_selfattn",
                             static_chunk_size=1)
    return TransformerLM(text_encoder_input_size=c["text_in"], llm_input_size=c["width"], llm_output_size=c["width"],
                         text_token_size=c["text_vocab"], speech_token_size=c["speech_tokens"], text_encoder=text_encoder,
                         llm=llm, sampling=ras_sampling, length_normalized_loss=True, lsm_weight=0.0,
                         spk_embed_dim=c["spk"])


def run_language_model():
    gen = torch.Generator().manual_seed(11)
    model = language_model()
    randomize(model, gen, 0.3)
    initial = flat(model)
    text = torch.randint(0, LM["text_vocab"], (1, 5), generator=gen)
    speech = torch.randint(0, LM["speech_tokens"], (1, 7), generator=gen)
    embedding = torch.randn(1, LM["spk"], generator=gen)
    batch = {"text_token": text, "text_token_len": torch.tensor([5]), "speech_token": speech,
             "speech_token_len": torch.tensor([7]), "embedding": embedding}
    model.train()
    loss = model(batch, torch.device("cpu"))["loss"]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"config": LM, "initial": initial, "text": text.flatten().tolist(), "speech": speech.flatten().tolist(),
            "embedding": embedding.flatten().tolist(), "loss": loss.item(), "lr": 0.1, "final": flat(model)}


FLOW = dict(vocab=10, input=8, mel=80, spk=6, heads=2, units=12, blocks=2, channels=[16, 16], head_dim=4,
            est_heads=2, est_blocks=2, est_mid=2)


def _sinusoidal_float64(self, x, scale=1000):
    # matcha's SinusoidalPosEmb with its frequencies in float64 instead of float32.
    if x.ndim < 1:
        x = x.unsqueeze(0)
    half_dim = self.dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float64) * -emb)
    emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat((emb.sin(), emb.cos()), dim=-1)


def flow_model():
    import types
    from cosyvoice.flow.decoder import ConditionalDecoder
    from cosyvoice.flow.flow import MaskedDiffWithXvec
    from cosyvoice.flow.flow_matching import ConditionalCFM
    from cosyvoice.flow.length_regulator import InterpolateRegulator
    from matcha.models.components import decoder as matcha_decoder
    matcha_decoder.SinusoidalPosEmb.forward = _sinusoidal_float64
    # mask_to_bias asserts a float32/16 dtype; the same bias in float64.
    from cosyvoice.flow import decoder as cv_decoder
    cv_decoder.mask_to_bias = lambda mask, dtype: (1.0 - mask.to(dtype)) * -1.0e+10
    c = FLOW
    encoder = ConformerEncoder(input_size=c["input"], output_size=c["input"], attention_heads=c["heads"],
                               linear_units=c["units"], num_blocks=c["blocks"], dropout_rate=0.0,
                               positional_dropout_rate=0.0, attention_dropout_rate=0.0, normalize_before=True,
                               input_layer="linear", pos_enc_layer_type="rel_pos_espnet",
                               selfattention_layer_type="rel_selfattn", use_cnn_module=False, macaron_style=False)
    regulator = InterpolateRegulator(channels=c["mel"], sampling_ratios=[1, 1, 1, 1])
    estimator = ConditionalDecoder(in_channels=4 * c["mel"], out_channels=c["mel"], channels=c["channels"], dropout=0.0,
                                   attention_head_dim=c["head_dim"], n_blocks=c["est_blocks"],
                                   num_mid_blocks=c["est_mid"], num_heads=c["est_heads"], act_fn="gelu")
    cfm_params = types.SimpleNamespace(sigma_min=1e-6, solver="euler", t_scheduler="cosine", training_cfg_rate=0.2,
                                       inference_cfg_rate=0.7, reg_loss_type="l1")
    decoder = ConditionalCFM(in_channels=3 * c["mel"], cfm_params=cfm_params, n_spks=1, spk_emb_dim=c["mel"],
                             estimator=estimator)
    return MaskedDiffWithXvec(input_size=c["input"], output_size=c["mel"], spk_embed_dim=c["spk"], output_type="mel",
                              vocab_size=c["vocab"], input_frame_rate=50, only_mask_loss=True, encoder=encoder,
                              length_regulator=regulator, decoder=decoder)


def run_flow():
    import copy
    import random as pyrandom
    from cosyvoice.flow import flow as cv_flow
    gen = torch.Generator().manual_seed(21)
    model = flow_model()
    randomize(model, gen, 0.2)
    initial = flat(model)
    c = FLOW
    tokens = torch.randint(0, c["vocab"], (1, 6), generator=gen)
    frames = 13
    feat = torch.randn(1, frames, c["mel"], generator=gen)
    embedding = torch.randn(1, c["spk"], generator=gen)
    prompt = 3
    t = torch.full((1, 1, 1), 0.37)
    z = torch.randn(1, c["mel"], frames, generator=gen)
    keep = torch.tensor([0.9])                          # > training_cfg_rate: the conditions are kept
    batch = {"speech_token": tokens, "speech_token_len": torch.tensor([6]), "speech_feat": feat,
             "speech_feat_len": torch.tensor([frames]), "embedding": embedding}

    captured = {}

    def capture(module, args, output):
        captured["pred"] = output

    def loss_and_step(net, absolute):
        rand_values = iter([t, keep])
        hook = net.decoder.estimator.register_forward_hook(capture)
        with mock.patch.object(cv_flow.random, "random", return_value=0.9),                 mock.patch.object(cv_flow.random, "randint", return_value=prompt),                 mock.patch("torch.rand", side_effect=lambda *a, **k: next(rand_values)),                 mock.patch("torch.randn_like", return_value=z.clone()):
            net.train()
            squared = net(batch, torch.device("cpu"))["loss"]
        hook.remove()
        if absolute:
            u = feat.transpose(1, 2) - (1 - 1e-6) * z
            loss = (captured["pred"] - u).abs().sum() / (frames * c["mel"])
        else:
            loss = squared
        optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), flat(net)

    squared_loss, squared_final = loss_and_step(copy.deepcopy(model), absolute=False)
    absolute_loss, absolute_final = loss_and_step(copy.deepcopy(model), absolute=True)

    def generate(token_count, prompt_tokens, prompt_frames, seed):
        g = torch.Generator().manual_seed(seed)
        text = torch.randint(0, c["vocab"], (1, token_count), generator=g)
        ptok = torch.randint(0, c["vocab"], (1, prompt_tokens), generator=g)
        pfeat = torch.randn(1, prompt_frames, c["mel"], generator=g)
        frames_out = prompt_frames + int(token_count / 50 * 22050 / 256)
        noise = torch.randn(1, c["mel"], frames_out, generator=g)
        net = copy.deepcopy(model)
        net.eval()
        with mock.patch("torch.randn_like", return_value=noise.clone()):
            mel, _ = net.inference(token=text, token_len=torch.tensor([token_count]), prompt_token=ptok,
                                   prompt_token_len=torch.tensor([prompt_tokens]), prompt_feat=pfeat,
                                   prompt_feat_len=torch.tensor([prompt_frames]), embedding=embedding,
                                   flow_cache=torch.zeros(1, c["mel"], 0, 2))
        return {"tokens": text.flatten().tolist(), "prompt_tokens": ptok.flatten().tolist(),
                "prompt_mel": pfeat.flatten().tolist(), "prompt_frames": prompt_frames,
                "noise": noise.flatten().tolist(), "mel": mel.transpose(1, 2).flatten().tolist(),
                "frames": mel.shape[2]}

    return {"config": FLOW, "initial": initial, "tokens": tokens.flatten().tolist(), "frames": frames,
            "mel": feat.flatten().tolist(), "embedding": embedding.flatten().tolist(), "prompt": prompt, "t": 0.37,
            "noise": z.flatten().tolist(), "lr": 0.05,
            "squared": {"loss": squared_loss, "final": squared_final},
            "absolute": {"loss": absolute_loss, "final": absolute_final},
            "generate_short": generate(6, 3, 5, 31), "generate_long": generate(45, 2, 4, 32)}


def main():
    out = {"cosyvoice_commit": COSYVOICE_COMMIT, "matcha_commit": MATCHA_COMMIT, "language_model": run_language_model(),
           "flow": run_flow()}
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "cosyvoice_reference.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("wrote", os.path.normpath(path), "lm loss", out["language_model"]["loss"], "flow", out["flow"]["squared"]["loss"], out["flow"]["absolute"]["loss"])


if __name__ == "__main__":
    main()
