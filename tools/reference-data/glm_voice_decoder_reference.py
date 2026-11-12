"""GLM-4-Voice's speech decoder reference: THUDM/GLM-4-Voice's cosyvoice fork (MaskedDiffWithXvec over a
BlockConformerEncoder, the nearest InterpolateRegulator, ConditionalCFM, HiFTGenerator) and flow_inference.AudioDecoder.
Pass the GLM-4-Voice checkout with --glm4voice and its Matcha-TTS submodule (dd9105b3) with --matcha.

Builds a tiny randomly initialized flow (80 mel channels, 12.5 Hz tokens, attention blocks of 3) and HiFT (seeded), runs
them in float64 and records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/glm_voice_decoder_reference.json:
  - the training path: the projected x-vector and mu (tokens encoded, nearest-stretched to the mel's frames), the
    flow-matching loss (compute_loss's formula with its cosine-scheduled t, noise and condition dropout given), and the
    loss after one SGD step (lr 0.1) on every flow parameter;
  - MaskedDiffWithXvec.inference after a prompt (tokens and mel), the whole sequence stretched and sampled in 10 guided
    Euler steps from given noise;
  - streaming decodes of 30 tokens through AudioDecoder.token2wav (the fade, the held-back frames and HiFT's cached
    excitation): the web demo's blocks (25, then 5) and stream_inference's blocks of 10 (the paper's 0.8 s).
torch.randn_like is replaced: a mel-shaped draw (ConditionalCFM's noise) reads a fixed sequence, any other (SineGen's
noise) is zero; SineGen's random phases are zero. Float32 steps of the reference are made float64: the ESPnet position
table, Matcha's time-embedding frequencies and HiFT's Hann window. Default: verify; --write: regenerate.
"""
import argparse
import base64
import math
import sys
import types

import huggingface_hub
import numpy as np
import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture

huggingface_hub.cached_download = huggingface_hub.hf_hub_download  # diffusers 0.25 imports the removed name
from wenet_encoder_reference import patch_float64_positions

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/glm_voice_decoder_reference.json"
MEL = 80
CONFIG = dict(vocab_size=16, input_size=8, output_size=MEL, spk_embed_dim=6, input_frame_rate=12.5)
ENCODER = dict(output_size=8, attention_heads=2, linear_units=12, num_blocks=2, block_size=3)
DECODER = dict(channels=[8, 8], attention_head_dim=4, n_blocks=1, num_mid_blocks=1, num_heads=2)
HIFT = dict(in_channels=MEL, base_channels=16, nb_harmonics=2, sampling_rate=22050, nsf_alpha=0.1, nsf_sigma=0.003,
            nsf_voiced_threshold=10, upsample_rates=[8, 8], upsample_kernel_sizes=[16, 16],
            istft_params={"n_fft": 16, "hop_len": 4}, resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], source_resblock_kernel_sizes=[7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]], lrelu_slope=0.1, audio_limit=0.99)
F0_CHANNELS = 16
REGULATOR_STAGES = 2
TOKENS = [3, 7, 1, 12, 5, 9, 4]
FRAMES = 13
T = 1 - math.cos(0.4 * 0.5 * math.pi)
LEARNING_RATE = 0.1
PROMPT_TOKENS = [2, 8, 11, 6]
PROMPT_FRAMES = 20
STREAM = [(5 * i + 3) % 16 for i in range(30)]
NOISE_LENGTH = MEL * 256
NOISE = None


def noise_for(frames):
    return NOISE[:MEL * frames].reshape(1, MEL, frames).clone()


def stub_imports(root, matcha):
    sys.path.insert(0, matcha)
    sys.path.insert(0, root)
    utils = types.ModuleType("matcha.utils")
    utils.__path__ = []
    pylogger = types.ModuleType("matcha.utils.pylogger")
    pylogger.get_pylogger = lambda name=__name__: __import__("logging").getLogger(name)
    sys.modules["matcha.utils"] = utils
    sys.modules["matcha.utils.pylogger"] = pylogger
    with open(f"{root}/cosyvoice/utils/class_utils.py", encoding="utf-8") as handle:
        source = "\n".join(line for line in handle.read().splitlines()
                           if not line.startswith(("from cosyvoice.llm", "from cosyvoice.flow", "from cosyvoice.hifigan",
                                                   "from cosyvoice.cli")))
    class_utils = types.ModuleType("cosyvoice.utils.class_utils")
    sys.modules["cosyvoice.utils.class_utils"] = class_utils
    exec(compile(source.split("def get_model_type")[0], "class_utils.py", "exec"), class_utils.__dict__)
    import cosyvoice.transformer.embedding as embedding
    patch_float64_positions(embedding)
    import matcha.models.components.decoder as matcha_decoder

    def sinusoidal(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        step = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float64) * -step)
        emb = scale * x.unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)
    matcha_decoder.SinusoidalPosEmb.forward = sinusoidal
    # The noise: ConditionalCFM's mel-shaped draw reads the fixed sequence, SineGen's is zero; its phases are zero.
    original = torch.randn_like

    def randn_like(x, *args, **kwargs):
        if x.dim() == 3 and x.shape[1] == MEL and NOISE is not None:
            return noise_for(x.shape[2]).to(x.dtype)
        return torch.zeros_like(x)
    torch.randn_like = randn_like
    torch.manual_seed = lambda seed: None
    import cosyvoice.hifigan.generator as generator
    generator.Uniform = lambda low, high: types.SimpleNamespace(sample=lambda sample_shape: torch.zeros(sample_shape))
    return original


def build(state=None, hift_state=None):
    from omegaconf import DictConfig
    from cosyvoice.flow.decoder import ConditionalDecoder
    from cosyvoice.flow.flow import MaskedDiffWithXvec
    from cosyvoice.flow.flow_matching import ConditionalCFM
    from cosyvoice.flow.length_regulator import InterpolateRegulator
    from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor
    from cosyvoice.hifigan.generator import HiFTGenerator
    from cosyvoice.transformer.encoder import BlockConformerEncoder
    torch.random.manual_seed(2294)
    encoder = BlockConformerEncoder(input_size=CONFIG["input_size"], dropout_rate=0.1, positional_dropout_rate=0.1,
                                    attention_dropout_rate=0.1, normalize_before=True, input_layer="linear",
                                    pos_enc_layer_type="rel_pos_espnet", selfattention_layer_type="block_rel_selfattn",
                                    use_cnn_module=False, macaron_style=False, **ENCODER)
    regulator = InterpolateRegulator(channels=MEL, sampling_ratios=[1] * REGULATOR_STAGES)
    estimator = ConditionalDecoder(in_channels=4 * MEL, out_channels=MEL, dropout=0.0, act_fn="gelu", **DECODER)
    cfm = ConditionalCFM(in_channels=3 * MEL, n_spks=1, spk_emb_dim=MEL, estimator=estimator,
                         cfm_params=DictConfig({"sigma_min": 1e-06, "solver": "euler", "t_scheduler": "cosine",
                                                "training_cfg_rate": 0.2, "inference_cfg_rate": 0.7, "reg_loss_type": "l1"}))
    flow = MaskedDiffWithXvec(output_type="mel", only_mask_loss=True, encoder=encoder, length_regulator=regulator,
                              decoder=cfm, **CONFIG).eval()
    hift = HiFTGenerator(**HIFT, f0_predictor=ConvRNNF0Predictor(1, MEL, F0_CHANNELS)).eval()
    with torch.no_grad():
        for name, p in flow.named_parameters():
            if name.endswith("bias") or "norm" in name or name.endswith("block.1.weight") or "length_regulator.model.1" in name \
                    or "length_regulator.model.4" in name:
                p.add_(0.1 * torch.randn(p.shape))
        for name, p in hift.named_parameters():
            if name.endswith(".alpha"):
                p.copy_(0.5 + torch.rand(p.shape))
    if state is not None:
        flow.load_state_dict(state, strict=True)
        hift.load_state_dict(hift_state, strict=True)
    from scipy.signal import get_window
    hift = hift.double()
    hift.stft_window = torch.from_numpy(get_window("hann", HIFT["istft_params"]["n_fft"], fftbins=True))
    return flow.double(), hift


def pieces(flow, xvector):
    tokens = torch.tensor([TOKENS])
    embedding = flow.spk_embed_affine_layer(torch.nn.functional.normalize(xvector, dim=1))
    token = flow.input_embedding(torch.clamp(tokens, min=0))
    h, _ = flow.encoder(token, torch.tensor([len(TOKENS)]))
    h = flow.encoder_proj(h)
    mu, _ = flow.length_regulator(h, torch.tensor([FRAMES]))
    return embedding, mu.transpose(1, 2)


def loss(flow, target, mu, spks, cond, noise, drop):
    sigma_min = flow.decoder.sigma_min
    t = torch.tensor([[[T]]], dtype=torch.float64)
    y = (1 - (1 - sigma_min) * t) * noise + t * target
    u = target - (1 - sigma_min) * noise
    if drop:
        mu, spks, cond = mu * 0, spks * 0, cond * 0
    mask = torch.ones(1, 1, FRAMES, dtype=torch.float64)
    pred = flow.decoder.estimator(y, mask, mu, t.squeeze(), spks, cond)
    return torch.nn.functional.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])


def stream(flow, hift, root, blocks):
    sys.path.insert(0, root)
    import importlib.util
    spec = importlib.util.spec_from_file_location("flow_inference_module", f"{root}/flow_inference.py")
    # flow_inference imports hyperpyyaml and torchaudio for its constructor and I/O only.
    sys.modules.setdefault("hyperpyyaml", types.SimpleNamespace(load_hyperpyyaml=None))
    sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from collections import defaultdict
    decoder = module.AudioDecoder.__new__(module.AudioDecoder)
    decoder.device, decoder.flow, decoder.hift = "cpu", flow, hift
    decoder.mel_overlap_dict = defaultdict(lambda: None)
    decoder.hift_cache_dict = defaultdict(lambda: None)
    decoder.token_overlap_len = 5
    decoder.mel_overlap_len = int(decoder.token_overlap_len / flow.input_frame_rate * 22050 / 256)
    decoder.mel_window = np.hamming(2 * decoder.mel_overlap_len)
    decoder.mel_cache_len = 1
    decoder.source_cache_len = int(decoder.mel_cache_len * 256)
    # web_demo.py's loop over the generated tokens, without the LM (blocks [25, 50, 100, 150, 200]); flow_inference's
    # stream_inference is the same loop over blocks of the encoder's block size (10 in the release).
    embedding = torch.zeros(1, flow.spk_embed_affine_layer.in_features, dtype=torch.float64)
    prompt_feat = torch.zeros(1, 0, MEL, dtype=torch.float64)
    prompt_token = torch.zeros(1, 0, dtype=torch.int64)
    speeches, mels, start, index = [], [], 0, 0
    while start < len(STREAM):
        size = blocks[min(index, len(blocks) - 1)]
        index += 1
        tts_token = torch.tensor([STREAM[start:start + size]])
        finalize = start + size >= len(STREAM)
        if mels:
            prompt_feat = torch.cat(mels, dim=-1).transpose(1, 2)
        speech, mel = decoder.token2wav(tts_token, uuid="u", prompt_token=prompt_token, prompt_feat=prompt_feat,
                                        embedding=embedding, finalize=finalize)
        speeches.append(speech.squeeze())
        mels.append(mel)
        prompt_token = torch.cat((prompt_token, tts_token), dim=-1)
        start += size
    return torch.cat(speeches, dim=-1)


def run(flow, hift, root, xvector, target, cond, noise, prompt_mel):
    spks, mu = pieces(flow, xvector)
    plain = loss(flow, target, mu, spks, cond, noise, False)
    dropped = loss(flow, target, mu, spks, cond, noise, True)
    with torch.no_grad():
        inferred = flow.inference(token=torch.tensor([TOKENS]), token_len=torch.tensor([len(TOKENS)]),
                                  prompt_token=torch.tensor([PROMPT_TOKENS]), prompt_token_len=torch.tensor([len(PROMPT_TOKENS)]),
                                  prompt_feat=prompt_mel, prompt_feat_len=torch.tensor([PROMPT_FRAMES]), embedding=xvector)
    # Under inference_mode: on the CPU fade_in_out's .cpu() returns the flow's inference tensor itself (the release on
    # CUDA gets a copy), which it updates in place.
    with torch.inference_mode():
        speech = stream(flow, hift, root, [25, 50, 100, 150, 200]).clone()
        blocks_speech = stream(flow, hift, root, [10]).clone()
    parameters = list(flow.parameters())
    grads = torch.autograd.grad(plain, parameters, allow_unused=True)
    with torch.no_grad():
        for p, g in zip(parameters, grads):
            if g is not None:
                p -= LEARNING_RATE * g
    spks_after, mu_after = pieces(flow, xvector)
    after = loss(flow, target, mu_after, spks_after, cond, noise, False)
    return spks, mu, plain, dropped, inferred, speech, blocks_speech, after


def main():
    global NOISE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glm4voice", required=True)
    parser.add_argument("--matcha", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    original_randn_like = stub_imports(args.glm4voice, args.matcha)
    shapes = {"xvector": (1, CONFIG["spk_embed_dim"]), "target": (1, MEL, FRAMES), "cond": (1, MEL, FRAMES),
              "noise": (1, MEL, FRAMES), "sequence": (NOISE_LENGTH,), "prompt_mel": (1, PROMPT_FRAMES, MEL)}
    if args.write:
        flow, hift = build()
        state = {k: v.detach().clone().contiguous().float() for k, v in flow.state_dict().items()}
        hift_state = {k: v.detach().clone().contiguous().float() for k, v in hift.state_dict().items()}
        flow, hift = build(state, hift_state)
        generator = torch.Generator().manual_seed(7)
        tensors = {k: torch.randn(s, generator=generator).float().double() for k, s in shapes.items()}
        tensors["cond"][:, :, FRAMES // 2:] = 0
    else:
        data = _fixture.load(FIXTURE)
        state = load_safetensors(base64.b64decode(data["flow_safetensors_base64"]))
        hift_state = load_safetensors(base64.b64decode(data["hift_safetensors_base64"]))
        flow, hift = build(state, hift_state)
        tensors = {k: torch.tensor(data[k], dtype=torch.float64).reshape(s) for k, s in shapes.items()}
    NOISE = tensors["sequence"]
    spks, mu, plain, dropped, inferred, speech, blocks_speech, after = run(flow, hift, args.glm4voice, tensors["xvector"], tensors["target"],
                                                            tensors["cond"], tensors["noise"], tensors["prompt_mel"])
    results = [("speaker", spks), ("mu", mu), ("inferred", inferred), ("speech", speech), ("blocks_speech", blocks_speech)]
    scalars = [("loss", plain), ("loss_dropped", dropped), ("loss_after_sgd", after)]
    if args.write:
        data = {"config": CONFIG, "encoder": ENCODER, "decoder": DECODER, "hift": HIFT, "f0_channels": F0_CHANNELS,
                "regulator_stages": REGULATOR_STAGES, "tokens": TOKENS, "frames": FRAMES, "t": T, "learning_rate": LEARNING_RATE,
                "prompt_tokens": PROMPT_TOKENS, "prompt_frames": PROMPT_FRAMES, "stream": STREAM,
                "flow_safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                "hift_safetensors_base64": base64.b64encode(save_safetensors(hift_state)).decode("ascii")}
        for k, v in tensors.items():
            data[k] = [float(x) for x in v.reshape(-1).tolist()]
        for k, v in results:
            data[k] = [float(x) for x in v.detach().reshape(-1).tolist()]
        for k, v in scalars:
            data[k] = float(v.detach())
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, "inferred", tuple(inferred.shape), "speech", tuple(speech.shape))
        return
    check = _fixture.Comparison("GLM-4-Voice decoder")
    for k, v in results:
        check.floats(k, data[k], v.detach().reshape(-1).tolist(), 1e-12, float)
    for k, v in scalars:
        check.floats(k, [data[k]], [float(v)], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
