# Reinforcement Learning Samples

This directory contains examples of RL agents in AiDotNet.

## Available Samples

| Sample | Description |
|--------|-------------|
| [CartPole](./CartPole/) | Classic cart-pole balancing |
| [DeepQLearning](./DeepQLearning/) | DQN for Atari-like games |

## Quick Start

```csharp
using AiDotNet;
using AiDotNet.ReinforcementLearning;

var agent = new PPOAgent<float>(
    stateSize: 4,
    actionSize: 2,
    hiddenSize: 64);

var env = new CartPoleEnvironment<float>();

for (int episode = 0; episode < 1000; episode++)
{
    var state = env.Reset();
    float totalReward = 0;

    while (!env.IsDone)
    {
        var action = agent.SelectAction(state);
        var (nextState, reward, done) = env.Step(action);
        agent.StoreTransition(state, action, reward, nextState, done);
        state = nextState;
        totalReward += reward;
    }

    agent.Update();
}
```

## RL Algorithms (80+)

### Value-Based
- DQN (Deep Q-Network)
- Double DQN
- Dueling DQN
- Rainbow

### Policy Gradient
- PPO (Proximal Policy Optimization)
- A2C/A3C
- TRPO
- REINFORCE

### Actor-Critic
- SAC (Soft Actor-Critic)
- DDPG
- TD3

### Multi-Agent
- MADDPG
- QMIX
- MAPPO

### Model-Based
- Dreamer
- MuZero
- World Models

## Learn More

- [RL Tutorial](/docs/tutorials/reinforcement-learning/)
- [API Reference](/api/AiDotNet.ReinforcementLearning/)
