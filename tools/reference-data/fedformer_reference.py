"""FEDformer reference: the official MAZiqing/FEDformer model (Zhou et al., ICML 2022) at a tiny size.

The network is the reference's own models/FEDformer.py (version 'Fourier'): frequency-enhanced blocks in the encoder
and decoder self-attention, frequency-enhanced cross-attention, series decomposition after every sublayer, and the
decoder's trend accumulation. Pass a checkout (commit c0f6b972) with --fedformer.

Three adjustments make it importable and exact, none of which changes the network:
- the wavelet module (imported but unused by the Fourier version) and matplotlib are stubbed when absent;
- FourierCorrelation.py allocates its spectra as torch.cfloat; it is executed with torch.cdouble instead, so the
  reference runs in float64 throughout, and its complex weights are cast to complex128;
- the calendar marks are zero, so the 'timeF' temporal embedding (a bias-free linear map of them) contributes nothing:
  the model is FEDformer on the values alone, which is how AiDotNet's FEDformer takes its input.

Modes are selected 'low' (deterministic); every parameter, the norms' included, is drawn at random (seeded).

The fixture records the configuration, the state dict (complex weights as their real and imaginary parts, under
'<name>.real' and '<name>.imag'), the frequency indices of every Fourier module, a batch of two input windows and the
forecast. Default: verify; --write: regenerate.
"""
import argparse
import importlib.util
import json
import os
import sys
import types

import torch

FIXTURE = "tests/AiDotNet.Tests/IntegrationTests/Finance/ReferenceData/fedformer_reference.json"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG = dict(seq_len=16, label_len=8, pred_len=4, enc_in=3, dec_in=3, c_out=3, d_model=16, n_heads=8, d_ff=32,
              e_layers=2, d_layers=1, modes=4, mode_select="low", moving_avg=5, dropout=0.0, activation="gelu",
              embed="timeF", freq="h", version="Fourier", output_attention=False)


def load_reference(root):
    sys.path.insert(0, root)
    if importlib.util.find_spec("matplotlib") is None:
        plt = types.ModuleType("matplotlib.pyplot")
        sys.modules["matplotlib"] = types.ModuleType("matplotlib")
        sys.modules["matplotlib.pyplot"] = plt
    wavelets = types.ModuleType("layers.MultiWaveletCorrelation")
    wavelets.MultiWaveletCross = wavelets.MultiWaveletTransform = object
    sys.modules["layers.MultiWaveletCorrelation"] = wavelets
    path = os.path.join(root, "layers", "FourierCorrelation.py")
    source = open(path, encoding="utf-8").read().replace("torch.cfloat", "torch.cdouble")
    fourier = types.ModuleType("layers.FourierCorrelation")
    fourier.__file__ = path
    exec(compile(source, path, "exec"), fourier.__dict__)
    sys.modules["layers.FourierCorrelation"] = fourier
    from models.FEDformer import Model
    return Model


def compute(root):
    Model = load_reference(root)
    torch.manual_seed(3)
    model = Model(types.SimpleNamespace(**CONFIG)).double().eval()
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "weights1"):
                module.weights1 = torch.nn.Parameter(module.weights1.data.to(torch.cdouble))
        for p in model.parameters():
            if p.is_complex():
                p.copy_(torch.complex(torch.randn(p.shape, generator=generator, dtype=torch.float64),
                                      torch.randn(p.shape, generator=generator, dtype=torch.float64)) * 0.05)
            else:
                p.copy_(torch.randn(p.shape, generator=generator, dtype=torch.float64) * 0.3)

    state, shapes = {}, {}
    for name, tensor in model.state_dict().items():
        if name.endswith('.position_embedding.pe'):
            continue  # a fixed sinusoid buffer the wo_pos embedding never reads
        if tensor.is_complex():
            for part in ("real", "imag"):
                values = getattr(tensor, part)
                state[f"{name}.{part}"] = values.reshape(-1).tolist()
                shapes[f"{name}.{part}"] = list(values.shape)
        else:
            state[name] = tensor.reshape(-1).tolist()
            shapes[name] = list(tensor.shape)
    indices = {}
    for name, module in model.named_modules():
        if hasattr(module, "index"):
            indices[name] = [int(i) for i in module.index]
        if hasattr(module, "index_q"):
            indices[name + ".q"] = [int(i) for i in module.index_q]
            indices[name + ".kv"] = [int(i) for i in module.index_kv]

    batch, marks = 2, 4
    x = torch.randn(batch, CONFIG["seq_len"], CONFIG["enc_in"], generator=generator, dtype=torch.float64)
    x_mark = torch.zeros(batch, CONFIG["seq_len"], marks, dtype=torch.float64)
    dec_len = CONFIG["label_len"] + CONFIG["pred_len"]
    x_dec = torch.zeros(batch, dec_len, CONFIG["dec_in"], dtype=torch.float64)
    x_mark_dec = torch.zeros(batch, dec_len, marks, dtype=torch.float64)
    with torch.no_grad():
        forecast = model(x, x_mark, x_dec, x_mark_dec)
    return dict(config=CONFIG, state_dict=state, shapes=shapes, indices=indices,
                input=x.reshape(-1).tolist(), input_shape=list(x.shape),
                forecast=forecast.reshape(-1).tolist(), forecast_shape=list(forecast.shape))


def max_deviation(expected, actual, what):
    """The largest elementwise difference; a length mismatch is a stale fixture, not a prefix to compare."""
    if len(expected) != len(actual):
        raise SystemExit(f"{what}: the fixture has {len(actual)} values, the reference {len(expected)}")
    return max((abs(a - b) for a, b in zip(expected, actual)), default=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fedformer", required=True, help="a checkout of MAZiqing/FEDformer at c0f6b972")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data = compute(os.path.abspath(args.fedformer))
    path = os.path.join(REPO, FIXTURE)
    if args.write:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"wrote {path}")
        return
    with open(path, encoding="utf-8") as f:
        stored = json.load(f)
    if data["config"] != stored["config"]:
        raise SystemExit("the fixture was written for another configuration")
    for key in ("input_shape", "forecast_shape"):
        if data[key] != stored[key]:
            raise SystemExit(f"{key}: the fixture has {stored[key]}, the reference {data[key]}")
    worst = max((max_deviation(values, stored["state_dict"].get(name, []), name)
                 for name, values in data["state_dict"].items()), default=0.0)
    worst = max(worst, max_deviation(data["input"], stored["input"], "input"))
    worst = max(worst, max_deviation(data["forecast"], stored["forecast"], "forecast"))
    print(f"max deviation from the fixture: {worst:.3e}")
    raise SystemExit(0 if worst <= 1e-12 else 1)


if __name__ == "__main__":
    main()
