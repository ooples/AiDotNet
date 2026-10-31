"""DeepAR reference: the network of Salinas et al. 2020 written directly in PyTorch.

The paper has no public reference code (it was built on Amazon's internal MXNet stack; GluonTS adds lags, time
features and its own scaler), so this transcribes §3 with PyTorch modules: an nn.LSTM (2 layers of 8 cells here) over
[z_{t-1}/nu, x_t], a linear head to the likelihood's raw parameters, the likelihood parameters mapped as the paper
states (Gaussian mu = o * nu, sigma = softplus(o) * nu; negative binomial mu = nu * softplus(o),
alpha = softplus(o) / sqrt(nu)) or as GluonTS states for its Student-t (df = 2 + softplus(o), loc = o * nu,
scale = softplus(o) * nu), and the mean negative log-likelihood over every step of the window (§3.2, t0 = 0) under
torch.distributions. nu is 1 + mean(z) over the conditioning range (§3.3), or mean |z| for the signed Student-t case.
For each likelihood the fixture records the weights, a window of 2 series (z0, then 12 steps with the first 8 as the
conditioning range) with 2 covariates, the loss, and the loss after one plain SGD step (lr 0.1). All in float64.
Default: verify; --write: regenerate tests/AiDotNet.Tests/UnitTests/TimeSeries/ReferenceData/deepar_reference.json.
"""
import argparse
import json
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/UnitTests/TimeSeries/ReferenceData/deepar_reference.json"
HIDDEN, LAYERS, COV, STEPS, CONDITIONING, BATCH = 8, 2, 2, 12, 8, 2


def compute():
    import torch
    import torch.nn.functional as F
    from torch.distributions import NegativeBinomial, Normal, StudentT
    torch.set_default_dtype(torch.float64)

    def case(likelihood, scaling, seed):
        torch.manual_seed(seed)
        outputs = 3 if likelihood == "StudentT" else 2
        lstm = torch.nn.LSTM(1 + COV, HIDDEN, LAYERS, batch_first=True)
        head = torch.nn.Linear(HIDDEN, outputs)
        g = torch.Generator().manual_seed(seed + 1)
        if likelihood == "NegativeBinomial":
            window = torch.poisson(torch.full((BATCH, STEPS + 1), 6.0), generator=g)
        elif scaling == "PaperMean":
            window = 3.0 + torch.rand(BATCH, STEPS + 1, generator=g) * 2
        else:
            window = torch.randn(BATCH, STEPS + 1, generator=g)
        window[:, 0] = window[:, 0] * (torch.arange(BATCH) > 0)          # series 0 starts at its first value: z0 = 0
        covariates = torch.randn(BATCH, STEPS, COV, generator=g)
        state = {f"lstm.{k}": v.detach().flatten().tolist() for k, v in lstm.state_dict().items()}
        state.update({f"head.{k}": v.detach().flatten().tolist() for k, v in head.state_dict().items()})
        params = list(lstm.parameters()) + list(head.parameters())

        def loss_of():
            cond = window[:, 1:1 + CONDITIONING]
            nu = 1 + cond.mean(dim=1) if scaling == "PaperMean" else cond.abs().mean(dim=1).clamp_min(1e-10)
            nu = nu.unsqueeze(1)
            inputs = torch.cat([(window[:, :-1] / nu).unsqueeze(-1), covariates], dim=-1)
            top, _ = lstm(inputs)
            raw = head(top)
            z = window[:, 1:]
            if likelihood == "Gaussian":
                dist = Normal(raw[..., 0] * nu, F.softplus(raw[..., 1]) * nu)
            elif likelihood == "NegativeBinomial":
                mu = nu * F.softplus(raw[..., 0])
                alpha = F.softplus(raw[..., 1]) / nu.sqrt()
                dist = NegativeBinomial(total_count=1 / alpha, probs=alpha * mu / (1 + alpha * mu))
            else:
                dist = StudentT(2 + F.softplus(raw[..., 0]), raw[..., 1] * nu, F.softplus(raw[..., 2]) * nu)
            return -dist.log_prob(z).mean()

        loss = loss_of()
        loss.backward()
        with torch.no_grad():
            for p in params:
                p -= 0.1 * p.grad
            after = loss_of()
        return {"likelihood": likelihood, "scaling": scaling, "state": state, "window": window.flatten().tolist(),
                "covariates": covariates.flatten().tolist(), "loss": loss.item(), "loss_after_sgd": after.item()}

    return {"config": dict(hidden=HIDDEN, layers=LAYERS, covariates=COV, steps=STEPS, conditioning=CONDITIONING, batch=BATCH),
            "cases": [case("Gaussian", "PaperMean", 3), case("NegativeBinomial", "PaperMean", 5), case("StudentT", "MeanAbsolute", 7)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print("wrote", [(c["likelihood"], round(c["loss"], 9), round(c["loss_after_sgd"], 9)) for c in result["cases"]])
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = all(abs(a["loss"] - b["loss"]) < 1e-12 and abs(a["loss_after_sgd"] - b["loss_after_sgd"]) < 1e-12
               for a, b in zip(stored["cases"], result["cases"]))
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
