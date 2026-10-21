"""Reference fixture for FinancialSACAgent: Soft Actor-Critic gradient steps as Haarnoja et al. define them.

A float64 PyTorch port of one replay-buffer training run (three gradient steps) for three configurations:
  A. 2018a (ICML) defaults: V toward min Q(s,a) - alpha log pi(a|s); Q1, Q2 toward r + gamma (1 - d) Vbar(s');
     the policy on mean(alpha log pi(a~|s) - min Q(s, a~)); Vbar <- tau V + (1 - tau) Vbar. In Algorithm 1's order.
  B. 2018b defaults: Q1, Q2 toward r + gamma (1 - d)(min Qbar(s', a') - alpha log pi(a'|s')); the policy as above;
     alpha on J = -mean(alpha (log pi + Hbar)), stepped on alpha itself (Algorithm 1); Qbar by Polyak averaging.
  C. 2018b with the authors' code's choices: every step from one parameter snapshot (the 2018a code), the policy on
     the mean of the Q-functions and alpha stepped as log alpha (softlearning), and a state-independent log-std head.

The policy is a Gaussian over pre-squash actions, a = M tanh(mu + sigma eps), log sigma clipped to [-20, 2], with the
exact change of variables log(1 - tanh^2 u) = 2 (log 2 - u - softplus(-2u)). Every network is a ReLU MLP with a
linear output; every optimizer is torch.optim.Adam (Kingma and Ba, eps 1e-8). The agent's draws come from .NET's
seeded System.Random, ported below: the replay buffer samples indices with Next(count), and each standard normal is
Box-Muller over two NextDouble calls, sqrt(-2 log(1 - U1)) cos(2 pi U2), row-major over [batch, action].

Writes tests/AiDotNet.Tests/ReinforcementLearning/ReferenceData/sac_reference.json. Run with the reference venv.
"""
import json
import math
import os

import torch

torch.set_default_dtype(torch.float64)


class DotNetRandom:
    """System.Random(int seed): .NET's legacy seeded generator (Knuth's subtractive method)."""
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

    def next_double(self):
        return self._internal_sample() * (1.0 / self.MBIG)

    def next(self, max_value):
        return int(self.next_double() * max_value)

    def standard_normal(self):
        u1 = 1.0 - self.next_double()
        u2 = self.next_double()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def normals(self, n, a):
        return torch.tensor([[self.standard_normal() for _ in range(a)] for _ in range(n)])


def mlp_params(gen, sizes):
    layers = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        w = (torch.randn(fan_in, fan_out, generator=gen) * 0.5).requires_grad_()
        b = (torch.randn(fan_out, generator=gen) * 0.1).requires_grad_()
        layers.append((w, b))
    return layers


def mlp(layers, x):
    for i, (w, b) in enumerate(layers):
        x = x @ w + b
        if i < len(layers) - 1:
            x = torch.relu(x)
    return x


def params_of(layers):
    return [p for layer in layers for p in layer]


def flat(layers):
    return [v for w, b in layers for v in w.detach().flatten().tolist() + b.detach().tolist()]


def copy_layers(layers):
    return [(w.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()) for w, b in layers]


def run(case):
    gen = torch.Generator().manual_seed(case["torch_seed"])
    s, a, h = case["state_size"], case["action_size"], case["hidden"]
    head = 2 * a if case["state_head"] else a
    actor = mlp_params(gen, [s] + h + [head])
    q1 = mlp_params(gen, [s + a] + h + [1])
    q2 = mlp_params(gen, [s + a] + h + [1])
    v1 = case["variant"] == "v1"
    value = mlp_params(gen, [s] + h + [1]) if v1 else None
    logstd = None if case["state_head"] else torch.tensor(case["logstd"]).requires_grad_()
    initial = {"actor": flat(actor), "critic1": flat(q1), "critic2": flat(q2),
               "value": flat(value) if v1 else [], "logstd": case["logstd"] if logstd is not None else []}
    target_value = copy_layers(value) if v1 else None
    target_q1 = None if v1 else copy_layers(q1)
    target_q2 = None if v1 else copy_layers(q2)

    n_transitions = case["transitions"]
    states = torch.randn(n_transitions, s, generator=gen)
    actions = torch.tanh(torch.randn(n_transitions, a, generator=gen)) * case["max_position"]
    rewards = torch.randn(n_transitions, generator=gen)
    next_states = torch.randn(n_transitions, s, generator=gen)
    dones = [i == 3 for i in range(n_transitions)]

    auto = case["auto_alpha"]
    log_param = case["temperature"] == "log_alpha"
    temperature = None
    if auto:
        temperature = torch.tensor([case["initial_log_alpha"] if log_param else math.exp(case["initial_log_alpha"])]).requires_grad_()

    def alpha_value():
        if not auto:
            return case["alpha"]
        return (torch.exp(temperature) if log_param else temperature).item()

    lr = case["lr"]
    actor_params = params_of(actor) + ([logstd] if logstd is not None else [])
    opt_actor = torch.optim.Adam(actor_params, lr=lr, eps=1e-8)
    opt_q1 = torch.optim.Adam(params_of(q1), lr=lr, eps=1e-8)
    opt_q2 = torch.optim.Adam(params_of(q2), lr=lr, eps=1e-8)
    opt_v = torch.optim.Adam(params_of(value), lr=lr, eps=1e-8) if v1 else None
    opt_t = torch.optim.Adam([temperature], lr=lr, eps=1e-8) if auto else None

    m = case["max_position"]

    def policy(x):
        out = mlp(actor, x)
        mean = out[:, :a]
        raw = out[:, a:] if logstd is None else logstd.expand(x.shape[0], a)
        return mean, torch.clamp(raw, -20.0, 2.0)

    def sample(mean, log_sigma, eps):
        u = mean + torch.exp(log_sigma) * eps
        act = m * torch.tanh(u)
        gauss = (-0.5 * eps * eps - log_sigma).sum(1) - a * 0.5 * math.log(2 * math.pi)
        correction = (2.0 * (math.log(2.0) - u - torch.nn.functional.softplus(-2.0 * u))).sum(1)
        return act, gauss - correction - a * math.log(m)

    def q_min(qa, qb, x):
        return torch.min(mlp(qa, x), mlp(qb, x))[:, 0]

    def aggregate(x):
        if case["aggregate"] == "mean":
            return 0.5 * (mlp(q1, x) + mlp(q2, x))[:, 0]
        if case["aggregate"] == "first":
            return mlp(q1, x)[:, 0]
        return q_min(q1, q2, x)

    def step(opt, loss):
        opt.zero_grad()
        loss.backward()
        opt.step()

    agent_rng = DotNetRandom(case["seed"])
    replay_rng = DotNetRandom(case["seed"])
    gamma, tau, b = case["gamma"], case["tau"], case["batch"]
    target_entropy = -1.0 * a
    losses = []
    for _ in range(case["steps"]):
        idx = torch.tensor([replay_rng.next(n_transitions) for _ in range(b)])
        st, ac, rw, ns = states[idx], actions[idx], rewards[idx], next_states[idx]
        nd = torch.tensor([0.0 if dones[i] else 1.0 for i in idx.tolist()])
        alpha = alpha_value()

        def q_targets():
            with torch.no_grad():
                if v1:
                    nxt = mlp(target_value, ns)[:, 0]
                else:
                    mean_n, ls_n = policy(ns)
                    act_n, logp_n = sample(mean_n, ls_n, agent_rng.normals(b, a))
                    nxt = q_min(target_q1, target_q2, torch.cat([ns, act_n], 1)) - alpha * logp_n
                return (rw + gamma * nd * nxt)[:, None]

        def policy_targets(eps):
            with torch.no_grad():
                mean, ls = policy(st)
                act, logp = sample(mean, ls, eps)
                return (q_min(q1, q2, torch.cat([st, act], 1)) - alpha * logp)[:, None], logp

        def q_loss(qnet, y):
            return 0.5 * ((mlp(qnet, torch.cat([st, ac], 1)) - y) ** 2).mean()

        def policy_loss(eps):
            mean, ls = policy(st)
            act, logp = sample(mean, ls, eps)
            return (alpha * logp - aggregate(torch.cat([st, act], 1))).mean()

        def temperature_step(logp):
            c = (logp + target_entropy).mean().item()
            value_alpha = torch.exp(temperature) if log_param else temperature
            step(opt_t, -(value_alpha.sum()) * c)

        if case["simultaneous"]:
            eps = agent_rng.normals(b, a)
            v_target, logp_snapshot = policy_targets(eps)
            y = q_targets()
            lp = policy_loss(eps)
            step(opt_actor, lp)
            l1 = q_loss(q1, y)
            step(opt_q1, l1)
            l2 = q_loss(q2, y)
            step(opt_q2, l2)
            if v1:
                step(opt_v, 0.5 * ((mlp(value, st) - v_target) ** 2).mean())
            if auto:
                temperature_step(logp_snapshot)
        elif v1:
            v_target, _ = policy_targets(agent_rng.normals(b, a))
            step(opt_v, 0.5 * ((mlp(value, st) - v_target) ** 2).mean())
            y = q_targets()
            l1 = q_loss(q1, y)
            step(opt_q1, l1)
            l2 = q_loss(q2, y)
            step(opt_q2, l2)
            lp = policy_loss(agent_rng.normals(b, a))
            step(opt_actor, lp)
            if auto:
                temperature_step(policy_targets(agent_rng.normals(b, a))[1])
        else:
            y = q_targets()
            l1 = q_loss(q1, y)
            step(opt_q1, l1)
            l2 = q_loss(q2, y)
            step(opt_q2, l2)
            lp = policy_loss(agent_rng.normals(b, a))
            step(opt_actor, lp)
            if auto:
                temperature_step(policy_targets(agent_rng.normals(b, a))[1])

        with torch.no_grad():
            pairs = [(value, target_value)] if v1 else [(q1, target_q1), (q2, target_q2)]
            for source, target in pairs:
                for (ws, bs), (wt, bt) in zip(source, target):
                    wt.copy_(tau * ws + (1 - tau) * wt)
                    bt.copy_(tau * bs + (1 - tau) * bt)
        losses.append(0.5 * (l1.item() + l2.item()) + lp.item())

    return {
        "case": case,
        "initial": initial,
        "states": states.flatten().tolist(),
        "actions": actions.flatten().tolist(),
        "rewards": rewards.tolist(),
        "next_states": next_states.flatten().tolist(),
        "dones": dones,
        "losses": losses,
        "final": {
            "actor": flat(actor), "critic1": flat(q1), "critic2": flat(q2),
            "value": flat(value) if v1 else [], "target_value": flat(target_value) if v1 else [],
            "target_critic1": [] if v1 else flat(target_q1), "target_critic2": [] if v1 else flat(target_q2),
            "logstd": logstd.detach().tolist() if logstd is not None else [],
            "temperature": temperature.detach().tolist() if temperature is not None else [],
        },
    }


def main():
    base = dict(state_size=3, action_size=2, hidden=[4], transitions=7, batch=5, steps=3, lr=1e-2, gamma=0.99,
                tau=0.05, max_position=1.5, seed=11, initial_log_alpha=-0.3, logstd=[0.2, -0.4])
    cases = [
        dict(base, name="v1", variant="v1", state_head=True, auto_alpha=False, alpha=0.3, temperature="alpha",
             aggregate="min", simultaneous=False, torch_seed=3),
        dict(base, name="v2", variant="v2", state_head=True, auto_alpha=True, alpha=0.0, temperature="alpha",
             aggregate="min", simultaneous=False, torch_seed=4),
        dict(base, name="v2_code", variant="v2", state_head=False, auto_alpha=True, alpha=0.0, temperature="log_alpha",
             aggregate="mean", simultaneous=True, torch_seed=5),
    ]
    out = {case["name"]: run(case) for case in cases}
    path = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "AiDotNet.Tests", "ReinforcementLearning",
                        "ReferenceData", "sac_reference.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("wrote", os.path.normpath(path), {k: [round(x, 6) for x in v["losses"]] for k, v in out.items()})


if __name__ == "__main__":
    main()
