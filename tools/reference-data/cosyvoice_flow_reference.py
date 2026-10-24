"""CosyVoice 1 flow reference: FunAudioLLM/CosyVoice's MaskedDiffWithXvec (cosyvoice/flow/flow.py) with its Conformer
encoder, InterpolateRegulator and ConditionalCFM over the ConditionalDecoder estimator. Pass checkouts with --cosyvoice
and --matcha (Matcha-TTS at CosyVoice's pinned commit).

Builds a tiny randomly initialized flow (seeded; 80 mel channels, which solve_euler hard-codes), runs it in float64 and
records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/cosyvoice_flow_reference.json:
  - the projected x-vector and mu, the encoded speech tokens stretched to the mel's length (the training path);
  - the flow-matching loss (compute_loss's formula, with its t, noise and condition dropout given rather than drawn),
    with and without the conditions dropped;
  - a 3-step guided Euler sample (solve_euler over the cosine schedule) from given noise;
  - the loss after one SGD step (lr 0.1) on every flow parameter.
Float32 steps of the reference are made float64: the ESPnet position table, Matcha's time-embedding frequencies and
mask_to_bias's dtype check. Default: verify; --write: regenerate.
"""
import argparse
import base64
import math
import sys
import types

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture
from wenet_encoder_reference import patch_float64_positions

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/cosyvoice_flow_reference.json"
MEL = 80
CONFIG = dict(vocab_size=16, input_size=8, output_size=MEL, spk_embed_dim=6, input_frame_rate=50)
ENCODER = dict(output_size=8, attention_heads=2, linear_units=12, num_blocks=2)
DECODER = dict(channels=[8, 8], attention_head_dim=4, n_blocks=1, num_mid_blocks=1, num_heads=2)
REGULATOR_STAGES = 2
TOKENS = [3, 7, 1, 12, 5, 9]
FRAMES = 8
T = 0.4
STEPS = 3
LEARNING_RATE = 0.1


def stub_imports(cosyvoice_root, matcha_root):
    sys.path.insert(0, matcha_root)
    sys.path.insert(0, cosyvoice_root)
    onnx_stub = types.ModuleType("cosyvoice.utils.onnx")
    onnx_stub.SpeechTokenExtractor = onnx_stub.online_feature = onnx_stub.onnx_path = None
    sys.modules.setdefault("cosyvoice.utils.onnx", onnx_stub)
    # matcha.utils imports hydra for its instantiators; the flow needs only its logger.
    utils = types.ModuleType("matcha.utils")
    utils.__path__ = []
    pylogger = types.ModuleType("matcha.utils.pylogger")
    pylogger.get_pylogger = lambda name=__name__: __import__("logging").getLogger(name)
    sys.modules["matcha.utils"] = utils
    sys.modules["matcha.utils.pylogger"] = pylogger
    with open(f"{cosyvoice_root}/cosyvoice/utils/class_utils.py", encoding="utf-8") as handle:
        source = "\n".join(line for line in handle.read().splitlines()
                           if not line.startswith(("from cosyvoice.llm", "from cosyvoice.flow", "from cosyvoice.hifigan",
                                                   "from cosyvoice.cli")))
    class_utils = types.ModuleType("cosyvoice.utils.class_utils")
    sys.modules["cosyvoice.utils.class_utils"] = class_utils
    exec(compile(source.split("def get_model_type")[0], "class_utils.py", "exec"), class_utils.__dict__)
    import cosyvoice.transformer.embedding as embedding
    patch_float64_positions(embedding)
    import cosyvoice.flow.decoder as decoder

    def mask_to_bias(mask, dtype):
        assert mask.dtype == torch.bool
        return (1.0 - mask.to(dtype)) * -1.0e+10
    decoder.mask_to_bias = mask_to_bias

    def sinusoidal(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        step = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float64) * -step)
        emb = scale * x.unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)
    decoder.SinusoidalPosEmb.forward = sinusoidal


def build(state=None):
    from omegaconf import DictConfig
    from cosyvoice.flow.decoder import ConditionalDecoder
    from cosyvoice.flow.flow import MaskedDiffWithXvec
    from cosyvoice.flow.flow_matching import ConditionalCFM
    from cosyvoice.flow.length_regulator import InterpolateRegulator
    from cosyvoice.transformer.encoder import ConformerEncoder
    torch.manual_seed(2294)
    encoder = ConformerEncoder(input_size=CONFIG["input_size"], dropout_rate=0.1, positional_dropout_rate=0.1,
                               attention_dropout_rate=0.1, normalize_before=True, input_layer="linear",
                               pos_enc_layer_type="rel_pos_espnet", selfattention_layer_type="rel_selfattn",
                               use_cnn_module=False, macaron_style=False, **ENCODER)
    regulator = InterpolateRegulator(channels=MEL, sampling_ratios=[1] * REGULATOR_STAGES)
    estimator = ConditionalDecoder(in_channels=4 * MEL, out_channels=MEL, dropout=0.0, act_fn="gelu", **DECODER)
    cfm = ConditionalCFM(in_channels=3 * MEL, n_spks=1, spk_emb_dim=MEL, estimator=estimator,
                         cfm_params=DictConfig({"sigma_min": 1e-06, "solver": "euler", "t_scheduler": "cosine",
                                                "training_cfg_rate": 0.2, "inference_cfg_rate": 0.7, "reg_loss_type": "l1"}))
    flow = MaskedDiffWithXvec(output_type="mel", only_mask_loss=True, encoder=encoder, length_regulator=regulator,
                              decoder=cfm, **CONFIG).eval()
    with torch.no_grad():
        for name, p in flow.named_parameters():
            if name.endswith("bias") or "norm" in name or name.endswith("block.1.weight") or "length_regulator.model.1" in name \
                    or "length_regulator.model.4" in name:
                p.add_(0.1 * torch.randn_like(p))
    if state is not None:
        flow.load_state_dict(state, strict=True)
    return flow


def pieces(flow, xvector):
    tokens = torch.tensor([TOKENS])
    embedding = flow.spk_embed_affine_layer(torch.nn.functional.normalize(xvector, dim=1))
    token = flow.input_embedding(torch.clamp(tokens, min=0))
    h, _ = flow.encoder(token, torch.tensor([len(TOKENS)]))
    h = flow.encoder_proj(h)
    mu, _ = flow.length_regulator(h, torch.tensor([FRAMES]))
    return embedding, mu.transpose(1, 2)


def loss(flow, target, mu, spks, cond, noise, drop):
    # compute_loss with t, z and the condition dropout given.
    sigma_min = flow.decoder.sigma_min
    t = torch.tensor([[[T]]], dtype=torch.float64)
    y = (1 - (1 - sigma_min) * t) * noise + t * target
    u = target - (1 - sigma_min) * noise
    if drop:
        mu, spks, cond = mu * 0, spks * 0, cond * 0
    mask = torch.ones(1, 1, FRAMES, dtype=torch.float64)
    pred = flow.decoder.estimator(y, mask, mu, t.squeeze(), spks, cond)
    return torch.nn.functional.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])


def run(flow, xvector, target, cond, noise, start):
    spks, mu = pieces(flow, xvector)
    plain = loss(flow, target, mu, spks, cond, noise, False)
    dropped = loss(flow, target, mu, spks, cond, noise, True)
    with torch.no_grad():
        t_span = 1 - torch.cos(torch.linspace(0, 1, STEPS + 1, dtype=torch.float64) * 0.5 * torch.pi)
        mask = torch.ones(1, 1, FRAMES, dtype=torch.float64)
        reference = flow.decoder.solve_euler(start.clone(), t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)
        # solve_euler returns its result cast to float32; the same Euler loop in float64, which must round to it exactly.
        sample, t, dt = start.clone(), t_span[0], t_span[1] - t_span[0]
        w = flow.decoder.inference_cfg_rate
        for step in range(1, len(t_span)):
            guided = flow.decoder.estimator(sample, mask, mu, t.unsqueeze(0), spks, cond)
            free = flow.decoder.estimator(sample, mask, mu * 0, t.unsqueeze(0), spks * 0, cond * 0)
            sample = sample + dt * ((1.0 + w) * guided - w * free)
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        assert torch.equal(sample.float(), reference), "the float64 Euler loop does not round to the reference's sample"
    parameters = list(flow.parameters())
    grads = torch.autograd.grad(plain, parameters, allow_unused=True)
    with torch.no_grad():
        for p, g in zip(parameters, grads):
            if g is not None:
                p -= LEARNING_RATE * g
    spks_after, mu_after = pieces(flow, xvector)
    after = loss(flow, target, mu_after, spks_after, cond, noise, False)
    return spks, mu, plain, dropped, sample, after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosyvoice", required=True)
    parser.add_argument("--matcha", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    stub_imports(args.cosyvoice, args.matcha)
    if args.write:
        flow = build()
        state = {k: v.detach().clone().contiguous() for k, v in flow.state_dict().items()}
        torch.manual_seed(7)
        tensors = {"xvector": torch.randn(1, CONFIG["spk_embed_dim"]), "target": torch.randn(1, MEL, FRAMES),
                   "cond": torch.randn(1, MEL, FRAMES), "noise": torch.randn(1, MEL, FRAMES), "start": torch.randn(1, MEL, FRAMES)}
        tensors["cond"][:, :, FRAMES // 2:] = 0                                  # a prompt over the first half
        tensors = {k: v.double() for k, v in tensors.items()}
        spks, mu, plain, dropped, sample, after = run(flow.double(), **tensors)
        data = {"config": CONFIG, "encoder": ENCODER, "decoder": DECODER, "regulator_stages": REGULATOR_STAGES, "tokens": TOKENS,
                "frames": FRAMES, "t": T, "steps": STEPS, "learning_rate": LEARNING_RATE,
                "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii")}
        for k, v in tensors.items():
            data[k] = [float(x) for x in v.reshape(-1).tolist()]
        for k, v in [("speaker", spks), ("mu", mu), ("sample", sample)]:
            data[k] = [float(x) for x in v.detach().reshape(-1).tolist()]
        data.update(loss=float(plain.detach()), loss_dropped=float(dropped.detach()), loss_after_sgd=float(after.detach()))
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    flow = build(load_safetensors(base64.b64decode(data["safetensors_base64"]))).double()
    shapes = {"xvector": (1, CONFIG["spk_embed_dim"]), "target": (1, MEL, FRAMES), "cond": (1, MEL, FRAMES),
              "noise": (1, MEL, FRAMES), "start": (1, MEL, FRAMES)}
    tensors = {k: torch.tensor(data[k], dtype=torch.float64).reshape(s) for k, s in shapes.items()}
    spks, mu, plain, dropped, sample, after = run(flow, **tensors)
    check = _fixture.Comparison("CosyVoice flow")
    for k, v in [("speaker", spks), ("mu", mu), ("sample", sample)]:
        check.floats(k, data[k], v.detach().reshape(-1).tolist(), 1e-12, float)
    for k, v in [("loss", plain), ("loss_dropped", dropped), ("loss_after_sgd", after)]:
        check.floats(k, [data[k]], [float(v)], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
