"""Autoformer reference: the official thuml/Autoformer model (models/Autoformer.py, layers/AutoCorrelation.py,
layers/Autoformer_EncDec.py, layers/Embed.py).

A small configuration (3 channels, 24 input steps, 12 start tokens, 8 predicted steps, D = 16, 4 heads, 2 encoder
layers, 1 decoder layer, d_ff = 32, moving average 5, factor 1, timeF embedding of 4 hourly features, GELU, no
dropout) runs in float64 from float32 weights. The fixture records:
- the state dict, the inputs (encoder series and time features, decoder time features) and a target;
- in eval mode (per-series lags, time_delay_agg_inference): the forecast;
- in train mode (lags shared by the batch, time_delay_agg_training): the forecast, the MSE loss, one plain SGD step
  (lr 0.1) on every parameter, and the train-mode loss after it.
The decoder input is built inside the model from the decomposition of the encoder input, so x_dec is only shaped.
Pass an Autoformer checkout with --autoformer. Default: verify; --write: regenerate
tests/AiDotNet.Tests/UnitTests/TimeSeries/ReferenceData/autoformer_reference.json.
"""
import argparse
import json
import os
import sys
import types

FIXTURE = "tests/AiDotNet.Tests/UnitTests/TimeSeries/ReferenceData/autoformer_reference.json"
C, SEQ, LABEL, PRED, DIM, HEADS, ENC, DEC, FF, MOVING, FACTOR, MARKS = 3, 24, 12, 8, 16, 4, 2, 1, 32, 5, 1, 4


def compute(autoformer):
    sys.path.insert(0, autoformer)
    import torch
    from models.Autoformer import Model

    configs = types.SimpleNamespace(seq_len=SEQ, label_len=LABEL, pred_len=PRED, output_attention=False,
                                    moving_avg=MOVING, enc_in=C, dec_in=C, c_out=C, d_model=DIM, embed="timeF", freq="h",
                                    dropout=0.0, factor=FACTOR, n_heads=HEADS, d_ff=FF, activation="gelu",
                                    e_layers=ENC, d_layers=DEC)
    torch.manual_seed(11)
    model = Model(configs)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.float())
    model = model.double()

    g = torch.Generator().manual_seed(5)
    t = torch.arange(SEQ, dtype=torch.float64)
    x_enc = (torch.sin(2 * torch.pi * t / 6).reshape(1, SEQ, 1) + 0.3 * torch.randn(2, SEQ, C, generator=g, dtype=torch.float64))
    mark_enc = torch.rand(2, SEQ, MARKS, generator=g, dtype=torch.float64) - 0.5
    mark_dec = torch.rand(2, LABEL + PRED, MARKS, generator=g, dtype=torch.float64) - 0.5
    target = torch.randn(2, PRED, C, generator=g, dtype=torch.float64)
    x_dec = torch.zeros(2, LABEL + PRED, C, dtype=torch.float64)
    state = {k: v.detach().float().flatten().tolist() for k, v in model.state_dict().items()}

    model.eval()
    with torch.no_grad():
        eval_forecast = model(x_enc, mark_enc, x_dec, mark_dec)
    model.train()
    forecast = model(x_enc, mark_enc, x_dec, mark_dec)
    loss = torch.nn.functional.mse_loss(forecast, target)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= 0.1 * p.grad
        after = torch.nn.functional.mse_loss(model(x_enc, mark_enc, x_dec, mark_dec), target)

    return {
        "config": dict(channels=C, seq_len=SEQ, label_len=LABEL, pred_len=PRED, d_model=DIM, n_heads=HEADS,
                       e_layers=ENC, d_layers=DEC, d_ff=FF, moving_avg=MOVING, factor=FACTOR, time_features=MARKS),
        "state": state,
        "x_enc": x_enc.flatten().tolist(),
        "mark_enc": mark_enc.flatten().tolist(),
        "mark_dec": mark_dec.flatten().tolist(),
        "target": target.flatten().tolist(),
        "eval_forecast": eval_forecast.flatten().tolist(),
        "train_forecast": forecast.detach().flatten().tolist(),
        "loss": loss.item(),
        "loss_after_sgd": after.item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--autoformer", default=os.path.expanduser("~/source/repos/ref/Autoformer"))
    args = parser.parse_args()
    result = compute(args.autoformer)
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print(f"wrote loss {result['loss']:.12f}, after SGD {result['loss_after_sgd']:.12f}")
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = abs(stored["loss"] - result["loss"]) < 1e-12 and abs(stored["loss_after_sgd"] - result["loss_after_sgd"]) < 1e-12
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
