"""Reference fixture for QMIXAgent: one QMIX training step (Rashid et al. 2018) as pymarl, the authors' code, runs it.

Runs pymarl's own QMixer (src/modules/mixers/qmix.py) and RNNAgent (src/modules/agents/rnn_agent.py), fetched at the
pinned commit, in float64, and ports QLearner.train: the agent network (shared across agents, fed the observation, the
last action one-hot and the agent id one-hot) unrolled over a batch of whole episodes from a zero hidden state; the
chosen-action Q-values and the target network's max over the next step's (or, with double Q, the target's value of
the live network's argmax); both mixed with the episode's states; y = r + gamma (1 - terminated) Qtot_target; the
squared TD error over filled steps; RMSprop (alpha 0.99, eps 1e-5) with the global gradient norm clipped at 10.

Two cases: the paper's configuration (single-layer hypernetworks, max over the target network, the summed squared TD
error of eq. 6) and pymarl's qmix.yaml (two-layer hypernetworks of 64, double Q, the masked mean). The batch has two
episodes: one of 3 steps that terminates and one of 2 steps that ends at the time limit (not terminated), padded.

Writes tests/AiDotNet.Tests/ReinforcementLearning/ReferenceData/qmix_reference.json. Run with the reference venv.
"""
import json
import math
import os
import types
import urllib.request

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

th.set_default_dtype(th.float64)
COMMIT = "c971afdceb34635d31b778021b0ef90d7af51e86"  # pymarl master when this fixture was generated
BASE = f"https://raw.githubusercontent.com/oxwhirl/pymarl/{COMMIT}/src/"


def official(path, name):
    with urllib.request.urlopen(BASE + path) as r:
        source = r.read().decode("utf-8")
    namespace = {}
    exec(compile(source, path, "exec"), namespace)
    return namespace[name]


QMixer = official("modules/mixers/qmix.py", "QMixer")
RNNAgent = official("modules/agents/rnn_agent.py", "RNNAgent")

N_AGENTS, OBS, ACTIONS, STATE, HIDDEN, EMBED = 2, 3, 3, 4, 5, 4
GAMMA, LR, CLIP = 0.99, 5e-4, 10.0


def flat(module):
    return {k: v.detach().flatten().tolist() for k, v in module.state_dict().items()}


def run(case):
    gen = th.Generator().manual_seed(case["seed"])
    args = types.SimpleNamespace(n_agents=N_AGENTS, state_shape=STATE, mixing_embed_dim=EMBED, rnn_hidden_dim=HIDDEN,
                                 n_actions=ACTIONS, hypernet_layers=case["hypernet_layers"], hypernet_embed=6)
    input_shape = OBS + ACTIONS + N_AGENTS
    agent, mixer = RNNAgent(input_shape, args), QMixer(args)
    with th.no_grad():
        for p in list(agent.parameters()) + list(mixer.parameters()):
            p.copy_(th.randn(p.shape, generator=gen) * 0.4)
    target_agent, target_mixer = RNNAgent(input_shape, args), QMixer(args)
    target_agent.load_state_dict(agent.state_dict())
    target_mixer.load_state_dict(mixer.state_dict())
    initial = {"agent": flat(agent), "mixer": flat(mixer)}

    lengths, terminated_last = [3, 2], [True, False]
    T = max(lengths)
    B = len(lengths)
    obs = th.randn(B, T + 1, N_AGENTS, OBS, generator=gen)
    state = th.randn(B, T + 1, STATE, generator=gen)
    actions = th.randint(0, ACTIONS, (B, T, N_AGENTS), generator=gen)
    reward = th.randn(B, T, generator=gen)
    terminated = th.zeros(B, T)
    filled = th.zeros(B, T)
    for b, length in enumerate(lengths):
        filled[b, :length] = 1
        if terminated_last[b]:
            terminated[b, length - 1] = 1
        # Padding past an episode's end: zeros, as an EpisodeBatch holds them.
        obs[b, length + 1:] = 0
        state[b, length + 1:] = 0
        actions[b, length:] = 0
        reward[b, length:] = 0
    onehot = F.one_hot(actions, ACTIONS).double()

    def unroll(net):
        h = th.zeros(B * N_AGENTS, HIDDEN)
        outs = []
        for t in range(T + 1):
            last = th.zeros(B, N_AGENTS, ACTIONS) if t == 0 else onehot[:, t - 1]
            ids = th.eye(N_AGENTS).unsqueeze(0).expand(B, -1, -1)
            inputs = th.cat([obs[:, t], last, ids], dim=2).reshape(B * N_AGENTS, -1)
            q, h = net(inputs, h)
            outs.append(q.view(B, N_AGENTS, ACTIONS))
        return th.stack(outs, dim=1)

    mask = filled.clone()
    mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
    mac_out = unroll(agent)
    chosen = th.gather(mac_out[:, :-1], 3, actions.unsqueeze(3)).squeeze(3)
    target_out = unroll(target_agent)[:, 1:]
    if case["double_q"]:
        best = mac_out.clone().detach()[:, 1:].max(dim=3, keepdim=True)[1]
        target_max = th.gather(target_out, 3, best).squeeze(3)
    else:
        target_max = target_out.max(dim=3)[0]
    q_tot = mixer(chosen, state[:, :-1]).squeeze(2)
    target_tot = target_mixer(target_max, state[:, 1:]).squeeze(2)
    targets = reward + GAMMA * (1 - terminated) * target_tot
    td = (q_tot - targets.detach()) * mask
    loss = (td ** 2).sum() / (mask.sum() if case["mean"] else 1.0)

    params = list(agent.parameters()) + list(mixer.parameters())
    optimiser = th.optim.RMSprop(params, lr=LR, alpha=0.99, eps=1e-5)
    optimiser.zero_grad()
    loss.backward()
    norm = th.nn.utils.clip_grad_norm_(params, CLIP)
    optimiser.step()
    return {
        "case": case,
        "initial": initial,
        "lengths": lengths,
        "terminated": terminated_last,
        "obs": obs.flatten().tolist(),
        "state": state.flatten().tolist(),
        "actions": actions.flatten().tolist(),
        "reward": reward.flatten().tolist(),
        "loss": loss.item(),
        "grad_norm": float(norm),
        "final": {"agent": flat(agent), "mixer": flat(mixer)},
    }


def main():
    cases = [
        dict(name="paper", hypernet_layers=1, double_q=False, mean=False, seed=3),
        dict(name="pymarl", hypernet_layers=2, double_q=True, mean=True, seed=4),
    ]
    out = {c["name"]: run(c) for c in cases}
    path = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "AiDotNet.Tests", "ReinforcementLearning",
                        "ReferenceData", "qmix_reference.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("wrote", os.path.normpath(path), {k: (v["loss"], v["grad_norm"]) for k, v in out.items()})


if __name__ == "__main__":
    main()
