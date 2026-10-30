"""CTC loss reference: torch.nn.functional.ctc_loss on fixed log-probabilities, for CTCLoss and the engine's TensorCTCLoss.

Records in tests/AiDotNet.Tests/UnitTests/LossFunctions/ReferenceData/ctc_loss_reference.json, in float64: log-softmax
outputs [time 9, batch 3, classes 5] (blank 0); targets with repeated labels (1 1 2), a single label (3) and a
repeat-heavy one (2 2 2 4); each sequence's negative log-likelihood (reduction "none") and the gradient of their sum
with respect to the logits under the log-softmax. (torch's gradient with respect to log-probabilities folds in the
softmax, exp(lp) − occupancy, which only agrees with the true −occupancy after the log-softmax.) Default: verify; --write: regenerate.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/UnitTests/LossFunctions/ReferenceData/ctc_loss_reference.json"
TARGETS = [[1, 1, 2], [3], [2, 2, 2, 4]]
INPUT_LENGTHS = [9, 7, 9]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    stored = None if args.write else _fixture.load(FIXTURE)
    if stored:
        logits = torch.tensor(stored["logits"], dtype=torch.float64).reshape(9, 3, 5)
    else:
        logits = torch.randn(9, 3, 5, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    logits = logits.detach().requires_grad_(True)
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = torch.tensor([label for target in TARGETS for label in target])
    losses = torch.nn.functional.ctc_loss(log_probs, targets, torch.tensor(INPUT_LENGTHS), torch.tensor([len(t) for t in TARGETS]),
                                          blank=0, reduction="none")
    losses.sum().backward()
    data = {"losses": [float(v) for v in losses.tolist()], "gradient": [float(v) for v in logits.grad.reshape(-1).tolist()],
            "log_probs": [float(v) for v in log_probs.detach().reshape(-1).tolist()]}
    if args.write:
        _fixture.save(FIXTURE, {"logits": [float(v) for v in logits.detach().reshape(-1).tolist()], "targets": TARGETS, "input_lengths": INPUT_LENGTHS, **data},
                      __file__)
        print("wrote", FIXTURE, data["losses"])
        return
    check = _fixture.Comparison("CTC loss")
    for k, v in data.items():
        check.floats(k, stored[k], v, 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
