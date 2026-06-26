---
title: "Reinforcement Learning"
description: "Train RL agents with AiDotNet."
order: 7
section: "Tutorials"
---

Train reinforcement-learning agents through the `AiModelBuilder` facade: `ConfigureModel(agent)` supplies the agent, and `ConfigureReinforcementLearning(options)` runs the training loop against an environment.

## How It Works

An RL agent learns by interacting with an environment — taking actions, receiving rewards, and improving its policy. AiDotNet drives that loop for you: you provide the agent and an environment, and the facade trains over a number of episodes. The agent is an `IFullModel<T, Vector<T>, Vector<T>>` (state in, action scores out), so the builder is typed `<T, Vector<T>, Vector<T>>`.

## PPO on CartPole

```csharp
using AiDotNet;
using AiDotNet.Configuration;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.PPO;
using AiDotNet.ReinforcementLearning.Environments;
using AiDotNet.Tensors.LinearAlgebra;

const int stateSize = 4;    // cart position/velocity, pole angle/angular velocity
const int actionSize = 2;   // push left / push right

var env = new CartPoleEnvironment<double>(maxSteps: 200, seed: 42);

var agent = new PPOAgent<double>(new PPOOptions<double>
{
    StateSize = stateSize,
    ActionSize = actionSize,
    PolicyLearningRate = 3e-4,
    ValueLearningRate = 1e-3,
    DiscountFactor = 0.99,
    GaeLambda = 0.95,
    ClipEpsilon = 0.2,
    PolicyHiddenLayers = new List<int> { 64, 64 }
});

var rlOptions = new RLTrainingOptions<double>
{
    Environment = env,
    Episodes = 25,
    MaxStepsPerEpisode = 200,
    Seed = 42
};

var result = await new AiModelBuilder<double, Vector<double>, Vector<double>>()
    .ConfigureModel(agent)
    .ConfigureReinforcementLearning(rlOptions)
    .BuildAsync();

// The trained policy maps a state to action scores via result.Predict.
var state = env.Reset();
var action = result.Predict(state);
int best = 0;
for (int i = 1; i < action.Length; i++)
    if (action[i] > action[best]) best = i;
Console.WriteLine($"Chosen action for the initial state: {best}");
```

## Agents & Environments

| Component | Examples |
|:----------|:---------|
| On-policy agents | `PPOAgent<T>` |
| Off-policy agents | `DQNAgent<T>`, `SACAgent<T>` |
| Environments | `CartPoleEnvironment<T>` (and your own via the environment interface) |

Swap `PPOAgent` for another agent and pass its matching options (`PPOOptions`, `DQNOptions`, …) to `ConfigureModel(...)`; the `ConfigureReinforcementLearning(...)` call is unchanged.

## Monitoring Training

`RLTrainingOptions<T>` accepts an `OnEpisodeComplete` callback of type `Action<RLEpisodeMetrics<T>>` that fires after each episode. Add it to the `rlOptions` above to log a reward curve — the metrics expose `Episode`, `TotalReward`, `Steps`, and `AverageRewardRecent`:

```text
OnEpisodeComplete = m => Console.WriteLine($"Episode {m.Episode}: reward={m.TotalReward:F1}")
```

## Best Practices

1. **Start with PPO**: it is stable and a strong default for discrete-action tasks.
2. **Tune the discount factor**: `DiscountFactor` near 0.99 balances short- and long-term reward.
3. **Right-size the policy network**: `PolicyHiddenLayers` like `[64, 64]` suit small control tasks.
4. **Use a fixed seed** while developing for reproducible episodes.
5. **Watch the reward curve** via `OnEpisodeComplete` to spot divergence early.

## Notes

The facade trains a single agent against an environment via `ConfigureReinforcementLearning`. Multi-agent systems (e.g. MADDPG), curriculum learning, and RL-specific hyperparameter tuners are not part of the facade surface today.

## Next Steps

- [CartPole Sample](/samples/reinforcement-learning/CartPole/)
- [Neural Network Training](/docs/examples/neural-network-training/)
