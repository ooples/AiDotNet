"""DeepAnT reference: the time-series predictor of Munir et al. (2019), as the PyTorch implementation at
github.com/datacubeR/DeepAnt (deepant.py, class DeepAnt) builds it. No author code was released; the paper fixes the
structure (two blocks of a 32-filter convolution, ReLU and max-pooling, a fully connected layer with ReLU and a linear
output of p_w values) and that implementation the widths the paper leaves out (kernel 3, pooling 2, 40 hidden units,
dropout 0.25). The module is transcribed below; its only change is the hidden layer's input width, which the
original hard-codes for a window of 10 (nn.Linear(32, 40)) and is computed here from the window.

A window of 15 (two pooled lengths of 6 and 2, so both poolings drop a trailing step) and p_w = 2 run in float64, in
eval mode (dropout off), from PyTorch's default initialization under a fixed seed. The fixture records:
- the state dict;
- three windows [3, 1, 15] and targets [3, 2];
- the prediction and the L1 loss (nn.L1Loss, the mean absolute error of the paper's Eq. 8);
- one plain SGD step (lr 0.1) on every parameter, and the loss after it.
Default: verify; --write: regenerate tests/AiDotNet.Tests/UnitTests/TimeSeries/ReferenceData/deepant_reference.json.
"""
import argparse
import json
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/UnitTests/TimeSeries/ReferenceData/deepant_reference.json"
SEQ_LEN, P_W = 15, 2


def compute():
    import torch
    import torch.nn as nn

    class DeepAnt(nn.Module):
        def __init__(self, seq_len, p_w):
            super().__init__()

            self.convblock1 = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding='valid'),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2)
            )

            self.convblock2 = nn.Sequential(
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding='valid'),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2)
            )

            self.flatten = nn.Flatten()

            flat = 32 * ((((seq_len - 2) // 2) - 2) // 2)
            self.denseblock = nn.Sequential(
                nn.Linear(flat, 40),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.25),
            )
            self.out = nn.Linear(40, p_w)

        def forward(self, x):
            x = self.convblock1(x)
            x = self.convblock2(x)
            x = self.flatten(x)
            x = self.denseblock(x)
            x = self.out(x)
            return x

    torch.manual_seed(11)
    model = DeepAnt(SEQ_LEN, P_W).double().eval()
    g = torch.Generator().manual_seed(5)
    windows = torch.rand(3, 1, SEQ_LEN, generator=g, dtype=torch.float64)
    targets = torch.rand(3, P_W, generator=g, dtype=torch.float64)

    state = {k: v.detach().flatten().tolist() for k, v in model.state_dict().items()}
    criterion = nn.L1Loss()
    prediction = model(windows)
    loss = criterion(prediction, targets)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= 0.1 * p.grad
        after = criterion(model(windows), targets)

    return {
        "config": dict(seq_len=SEQ_LEN, p_w=P_W, filters=32, kernel=3, pool=2, hidden=40),
        "state": state,
        "windows": windows.flatten().tolist(),
        "targets": targets.flatten().tolist(),
        "prediction": prediction.detach().flatten().tolist(),
        "loss": loss.item(),
        "loss_after_sgd": after.item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
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
