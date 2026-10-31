"""Reference fixture for FinancialPPOAgent: one PPO update (Schulman et al. 2017) as OpenAI Baselines' ppo1 runs it.

A float64 PyTorch port of pposgd_simple.learn's iteration body on a fixed rollout:
  - values and the bootstrap value under the observation statistics the actions were taken with (RunningMeanStd:
    sum 0, sum of squares and count 1e-2, variance floored at 1e-2, observations clipped to +-5);
  - add_vtarg_and_adv (GAE), advantages standardized with numpy's population std;
  - ob_rms.update(rollout), then oldpi = pi;
  - optim_epochs passes, each reshuffling the persistent order in place (Fisher-Yates driven by .NET's seeded
    System.Random, ported below, standing in for numpy's shuffle), minibatches of optim_batchsize with the final
    partial one dropped (Dataset.iterate_once);
  - loss = -mean(min(r A, clip(r) A)) - ent_coef mean(H) + vf_coef mean((V - ret)^2), one Adam (Kingma & Ba, as
    torch.optim.Adam) over the policy, the value network and the Gaussian's state-independent logstd, optionally
    clipping the global gradient norm (CleanRL's addition).

Two cases: a discrete policy with CleanRL's options (entropy 0.01, value weight 0.5, gradient norm 0.5) and a
Gaussian policy at the paper's defaults. Writes tests/AiDotNet.Tests/ReinforcementLearning/ReferenceData/ppo_reference.json.

Run with the reference venv: ~/source/repos/.venv-ttsref312/Scripts/python.exe tools/reference-data/ppo_reference.py
"""
import json
import math
import os

import torch

torch.set_default_dtype(torch.float64)


class DotNetRandom:
    """System.Random(int seed): .NET's legacy seeded generator (Knuth's subtractive method), Next(max) = (int)(Sample() * max)."""
    MBIG = 2147483647
    MSEED = 161803398

    def __init__(self, seed):
        self.seed_array = [0] * 56
        subtraction = 2147483647 if seed == -2147483648 else abs(seed)
        mj = self.MSEED - subtraction
        self.seed_array[55] = mj
        mk = 1
        for i in range(1, 55):
            ii = (21 * i) % 55
            self.seed_array[ii] = mk
            mk = mj - mk
            if mk < 0:
                mk += self.MBIG
            mj = self.seed_array[ii]
        for _ in range(4):
            for i in range(1, 56):
                self.seed_array[i] -= self.seed_array[1 + (i + 30) % 55]
                if self.seed_array[i] < 0:
                    self.seed_array[i] += self.MBIG
        self.inext = 0
        self.inextp = 21

    def _internal_sample(self):
        self.inext += 1
        if self.inext >= 56:
            self.inext = 1
        self.inextp += 1
        if self.inextp >= 56:
            self.inextp = 1
        value = self.seed_array[self.inext] - self.seed_array[self.inextp]
        if value == self.MBIG:
            value -= 1
        if value < 0:
            value += self.MBIG
        self.seed_array[self.inext] = value
        return value

    def next(self, max_value):
        return int(self._internal_sample() * (1.0 / self.MBIG) * max_value)


def mlp_params(gen, sizes):
    layers = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        w = (torch.randn(fan_in, fan_out, generator=gen) * 0.6).requires_grad_()
        b = (torch.randn(fan_out, generator=gen) * 0.2).requires_grad_()
        layers.append((w, b))
    return layers


def mlp(layers, x):
    for i, (w, b) in enumerate(layers):
        x = x @ w + b
        if i < len(layers) - 1:
            x = torch.tanh(x)
    return x


def normalize(x, total, squares, count, clip=5.0):
    mean = total / count
    std = torch.sqrt(torch.clamp(squares / count - mean * mean, min=1e-2))
    return torch.clamp((x - mean) / std, -clip, clip)


def log_prob_entropy(output, actions, logstd):
    if logstd is None:
        lse = torch.logsumexp(output, dim=1)
        chosen = (actions * output).sum(1)
        p = torch.softmax(output, dim=1)
        return chosen - lse, lse - (p * output).sum(1)
    a = actions.shape[1]
    z = (actions - output) / torch.exp(logstd)
    logp = -0.5 * (z * z).sum(1) - 0.5 * math.log(2 * math.pi) * a - logstd.sum()
    ent = (logstd + 0.5 * math.log(2 * math.pi * math.e)).sum().expand(output.shape[0])
    return logp, ent


def run(case):
    gen = torch.Generator().manual_seed(case["torch_seed"])
    s, a, t = case["state_size"], case["action_size"], case["rollout"]
    hidden = case["hidden"]
    actor = mlp_params(gen, [s] + hidden + [a])
    critic = mlp_params(gen, [s] + hidden + [1])
    logstd = torch.tensor(case["logstd"]).requires_grad_() if case["continuous"] else None

    states = torch.randn(t, s, generator=gen) * 2.5 + torch.tensor([1.0, -0.5, 6.0])[:s]
    next_last = torch.randn(s, generator=gen) * 2.5 + torch.tensor([1.0, -0.5, 6.0])[:s]
    if case["continuous"]:
        actions = torch.randn(t, a, generator=gen)
    else:
        actions = torch.nn.functional.one_hot(torch.randint(0, a, (t,), generator=gen), a).double()
    rewards = torch.randn(t, generator=gen)
    dones = [False] * t
    dones[4] = True

    total = torch.zeros(s)
    squares = torch.full((s,), 1e-2)
    count = 1e-2
    with torch.no_grad():
        values = mlp(critic, normalize(states, total, squares, count))[:, 0]
        bootstrap = 0.0 if dones[-1] else mlp(critic, normalize(next_last[None], total, squares, count))[0, 0].item()

    gamma, lam = case["gamma"], case["lambda"]
    adv = [0.0] * t
    last = 0.0
    for i in reversed(range(t)):
        nonterminal = 0.0 if dones[i] else 1.0
        nxt = bootstrap if i == t - 1 else values[i + 1].item()
        delta = rewards[i].item() + gamma * nxt * nonterminal - values[i].item()
        last = delta + gamma * lam * nonterminal * last
        adv[i] = last
    adv = torch.tensor(adv)
    returns = adv + values
    adv = (adv - adv.mean()) / adv.std(unbiased=False)

    total = total + states.sum(0)
    squares = squares + (states * states).sum(0)
    count = count + t
    x = normalize(states, total, squares, count)
    with torch.no_grad():
        old_logp, _ = log_prob_entropy(mlp(actor, x), actions, logstd)

    params = [p for layer in actor for p in layer] + [p for layer in critic for p in layer] + ([logstd] if logstd is not None else [])
    optimizer = torch.optim.Adam(params, lr=case["lr"], betas=(0.9, 0.999), eps=case["adam_eps"])
    rng = DotNetRandom(case["seed"])
    order = list(range(t))
    losses = []
    m = case["minibatch"]
    for _ in range(case["epochs"]):
        for i in range(t - 1, 0, -1):
            j = rng.next(i + 1)
            order[i], order[j] = order[j], order[i]
        for start in range(0, t - m + 1, m):
            idx = torch.tensor(order[start:start + m])
            logp, ent = log_prob_entropy(mlp(actor, x[idx]), actions[idx], logstd)
            ratio = torch.exp(logp - old_logp[idx])
            surrogate = torch.min(ratio * adv[idx], torch.clamp(ratio, 1 - case["clip"], 1 + case["clip"]) * adv[idx])
            value_loss = ((mlp(critic, x[idx])[:, 0] - returns[idx]) ** 2).mean()
            loss = -surrogate.mean() - case["ent_coef"] * ent.mean() + case["vf_coef"] * value_loss
            optimizer.zero_grad()
            loss.backward()
            if case["max_grad_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(params, case["max_grad_norm"])
            optimizer.step()
            losses.append(loss.item())

    def flat(layers):
        return [v for w, b in layers for v in w.detach().flatten().tolist() + b.detach().tolist()]

    probe = DotNetRandom(case["seed"])
    return {
        "case": case,
        "random_probe": [probe.next(1000) for _ in range(8)],
        "initial": case["initial"],
        "states": states.flatten().tolist(),
        "next_last": next_last.tolist(),
        "actions": actions.flatten().tolist(),
        "rewards": rewards.tolist(),
        "dones": dones,
        "mean_loss": sum(losses) / len(losses),
        "actor": flat(actor),
        "critic": flat(critic),
        "logstd": logstd.detach().tolist() if logstd is not None else [],
    }


def main():
    cases = [
        dict(name="discrete", continuous=False, state_size=3, action_size=3, hidden=[4], rollout=10, minibatch=4,
             epochs=3, lr=1e-2, adam_eps=1e-5, gamma=0.99, **{"lambda": 0.95}, clip=0.2, ent_coef=0.01, vf_coef=0.5,
             max_grad_norm=0.5, seed=11, torch_seed=3, logstd=None),
        dict(name="continuous", continuous=True, state_size=3, action_size=2, hidden=[4], rollout=10, minibatch=4,
             epochs=3, lr=1e-2, adam_eps=1e-5, gamma=0.99, **{"lambda": 0.95}, clip=0.2, ent_coef=0.0, vf_coef=1.0,
             max_grad_norm=None, seed=11, torch_seed=5, logstd=[0.1, -0.2]),
    ]
    out = {}
    for case in cases:
        # The initial weights, recorded before training mutates them.
        gen = torch.Generator().manual_seed(case["torch_seed"])
        s, a = case["state_size"], case["action_size"]
        actor = mlp_params(gen, [s] + case["hidden"] + [a])
        critic = mlp_params(gen, [s] + case["hidden"] + [1])
        flat = lambda layers: [v for w, b in layers for v in w.detach().flatten().tolist() + b.detach().tolist()]
        case["initial"] = {"actor": flat(actor), "critic": flat(critic)}
        out[case["name"]] = run(case)
    path = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "AiDotNet.Tests", "ReinforcementLearning",
                        "ReferenceData", "ppo_reference.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("wrote", os.path.normpath(path))


if __name__ == "__main__":
    main()
