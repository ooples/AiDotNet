"""Informer reference: the official zhouhaoyi/Informer2020 model (models/model.py, attn.py, encoder.py, decoder.py).

A small configuration (3 channels, 24 input steps, 12 start tokens, 6 predicted steps, D = 16, 4 heads, 3 encoder
layers with 2 distilling layers, 2 decoder layers, d_ff = 32, factor 2, timeF embedding of 4 hourly features, GELU,
mix, no dropout) runs in float64 from float32 weights, in eval mode. PositionalEmbedding builds its sinusoid table
in float32 (torch.zeros(...).float()) and .double() only widens the rounded values, which puts up to 2.3e-7 into
every embedding; the table is recomputed in float64 with the same formula. The fixture records:
- the state dict;
- the inputs (encoder series and time features, the decoder input [last 12 steps; zeros] and its time features) and
  a target;
- every torch.randint draw of ProbAttention (the sampled key indices), in call order, so the C# network replays them;
- with ProbSparse attention: the forecast and the MSE loss; then one plain SGD step (lr 0.1) on every parameter and
  the loss after it (whose forward draws the next indices);
- with full attention (attn='full', same weights): the forecast.
Pass an Informer2020 checkout with --informer. Default: verify; --write: regenerate
tests/AiDotNet.Tests/UnitTests/TimeSeries/ReferenceData/informer_reference.json.
"""
import argparse
import json
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/UnitTests/TimeSeries/ReferenceData/informer_reference.json"
C, SEQ, LABEL, PRED, DIM, HEADS, ENC, DEC, FF, FACTOR, MARKS = 3, 24, 12, 6, 16, 4, 3, 2, 32, 2, 4


def compute(informer):
    sys.path.insert(0, informer)
    import torch
    from models.model import Informer

    draws = []
    original = torch.randint
    generator = torch.Generator().manual_seed(1234)

    def recording_randint(high, size, **kwargs):
        out = original(high, size, generator=generator)
        draws.append(out.tolist())
        return out

    def build(attn):
        torch.manual_seed(7)
        model = Informer(C, C, C, SEQ, LABEL, PRED, factor=FACTOR, d_model=DIM, n_heads=HEADS, e_layers=ENC,
                         d_layers=DEC, d_ff=FF, dropout=0.0, attn=attn, embed="timeF", freq="h", activation="gelu",
                         distil=True, mix=True, device=torch.device("cpu"))
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(p.float())
        model = model.double().eval()
        # The position table in float64 (the reference computes it in float32).
        import math
        for embedding in (model.enc_embedding, model.dec_embedding):
            pe = embedding.position_embedding.pe
            position = torch.arange(0, pe.shape[1], dtype=torch.float64).unsqueeze(1)
            div_term = (torch.arange(0, DIM, 2, dtype=torch.float64) * -(math.log(10000.0) / DIM)).exp()
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        return model

    g = torch.Generator().manual_seed(3)
    x_enc = torch.randn(2, SEQ, C, generator=g, dtype=torch.float64)
    mark_enc = torch.rand(2, SEQ, MARKS, generator=g, dtype=torch.float64) - 0.5
    mark_dec = torch.rand(2, LABEL + PRED, MARKS, generator=g, dtype=torch.float64) - 0.5
    target = torch.randn(2, PRED, C, generator=g, dtype=torch.float64)
    x_dec = torch.cat([x_enc[:, -LABEL:, :], torch.zeros(2, PRED, C, dtype=torch.float64)], dim=1)

    model = build("prob")
    full = build("full")
    full.load_state_dict(model.state_dict())

    torch.randint = recording_randint
    try:
        forecast = model(x_enc, mark_enc, x_dec, mark_dec)
        loss = torch.nn.functional.mse_loss(forecast, target)
        loss.backward()
        with torch.no_grad():
            state = {k: v.detach().float().flatten().tolist() for k, v in model.state_dict().items()}
            for p in model.parameters():
                p -= 0.1 * p.grad
            after = torch.nn.functional.mse_loss(model(x_enc, mark_enc, x_dec, mark_dec), target)
    finally:
        torch.randint = original
    with torch.no_grad():
        full_forecast = full(x_enc, mark_enc, x_dec, mark_dec)

    return {
        "config": dict(channels=C, seq_len=SEQ, label_len=LABEL, pred_len=PRED, d_model=DIM, n_heads=HEADS,
                       e_layers=ENC, d_layers=DEC, d_ff=FF, factor=FACTOR, time_features=MARKS),
        "state": state,
        "x_enc": x_enc.flatten().tolist(),
        "mark_enc": mark_enc.flatten().tolist(),
        "x_dec": x_dec.flatten().tolist(),
        "mark_dec": mark_dec.flatten().tolist(),
        "target": target.flatten().tolist(),
        "draws": draws,
        "forecast": forecast.detach().flatten().tolist(),
        "loss": loss.item(),
        "loss_after_sgd": after.item(),
        "full_forecast": full_forecast.flatten().tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--informer", default=os.path.expanduser("~/source/repos/ref/Informer2020"))
    args = parser.parse_args()
    result = compute(args.informer)
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print(f"wrote loss {result['loss']:.12f}, after SGD {result['loss_after_sgd']:.12f}, {len(result['draws'])} draws")
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = (abs(stored["loss"] - result["loss"]) < 1e-12 and abs(stored["loss_after_sgd"] - result["loss_after_sgd"]) < 1e-12
            and stored["draws"] == result["draws"])
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
