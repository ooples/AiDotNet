---
layout: default
title: Reinforcement Learning
parent: Tutorials
nav_order: 6
has_children: true
permalink: /tutorials/reinforcement-learning/
---

# Reinforcement Learning Tutorial
{: .no_toc }

Train agents to make decisions through interaction with environments.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

AiDotNet provides 80+ RL algorithms:
- **Value-Based**: DQN, Double DQN, Dueling DQN, Rainbow
- **Policy Gradient**: PPO, A2C, TRPO, REINFORCE
- **Actor-Critic**: SAC, DDPG, TD3
- **Multi-Agent**: MADDPG, QMIX, MAPPO
- **Model-Based**: Dreamer, MuZero

---

## Basic Concepts

### The RL Loop

```csharp
var agent = new PPOAgent<float>(stateSize: 4, actionSize: 2);
var env = new CartPoleEnvironment<float>();

for (int episode = 0; episode < 1000; episode++)
{
    var state = env.Reset();
    float totalReward = 0;

    while (!env.IsDone)
    {
        // Agent chooses action
        var action = agent.SelectAction(state);

        // Environment responds
        var (nextState, reward, done) = env.Step(action);

        // Store experience
        agent.StoreTransition(state, action, reward, nextState, done);

        state = nextState;
        totalReward += reward;
    }

    // Learn from experiences
    agent.Update();

    Console.WriteLine($"Episode {episode}: Reward = {totalReward}");
}
```

---

## DQN (Deep Q-Network)

```csharp
using AiDotNet.ReinforcementLearning;

var dqnConfig = new DQNConfig<float>
{
    StateSize = 4,
    ActionSize = 2,
    HiddenLayers = [128, 128],
    LearningRate = 1e-3f,
    Gamma = 0.99f,
    EpsilonStart = 1.0f,
    EpsilonEnd = 0.01f,
    EpsilonDecay = 0.995f,
    ReplayBufferSize = 100000,
    BatchSize = 64,
    TargetUpdateFrequency = 100
};

var agent = new DQNAgent<float>(dqnConfig);

// Training loop
for (int episode = 0; episode < 500; episode++)
{
    var state = env.Reset();

    while (!env.IsDone)
    {
        var action = agent.SelectAction(state);
        var (nextState, reward, done) = env.Step(action);

        agent.StoreTransition(state, action, reward, nextState, done);
        agent.Update();

        state = nextState;
    }
}
```

---

## PPO (Proximal Policy Optimization)

Most popular algorithm for continuous control:

```csharp
var ppoConfig = new PPOConfig<float>
{
    StateSize = 8,
    ActionSize = 4,
    HiddenLayers = [256, 256],
    LearningRate = 3e-4f,
    Gamma = 0.99f,
    Lambda = 0.95f,         // GAE lambda
    ClipRatio = 0.2f,       // PPO clip parameter
    EntropyCoefficient = 0.01f,
    ValueCoefficient = 0.5f,
    MaxGradNorm = 0.5f,
    NumEpochs = 10,
    MiniBatchSize = 64,
    StepsPerUpdate = 2048
};

var agent = new PPOAgent<float>(ppoConfig);
```

---

## SAC (Soft Actor-Critic)

Best for continuous action spaces:

```csharp
var sacConfig = new SACConfig<float>
{
    StateSize = 11,
    ActionSize = 3,
    HiddenLayers = [256, 256],
    LearningRate = 3e-4f,
    Gamma = 0.99f,
    Tau = 0.005f,            // Soft update coefficient
    Alpha = 0.2f,            // Temperature parameter
    AutoTuneAlpha = true,    // Learn temperature
    ReplayBufferSize = 1000000,
    BatchSize = 256
};

var agent = new SACAgent<float>(sacConfig);
```

---

## Custom Environments

```csharp
public class MyEnvironment<T> : IEnvironment<T>
{
    public int StateSize => 4;
    public int ActionSize => 2;
    public bool IsDone { get; private set; }

    private T[] _state;

    public T[] Reset()
    {
        IsDone = false;
        _state = new T[StateSize];
        // Initialize state
        return _state;
    }

    public (T[] NextState, float Reward, bool Done) Step(int action)
    {
        // Apply action, update state
        float reward = ComputeReward();
        IsDone = CheckTermination();

        return (_state, reward, IsDone);
    }
}
```

---

## Multi-Agent RL

```csharp
using AiDotNet.ReinforcementLearning.MultiAgent;

var maddpgConfig = new MADDPGConfig<float>
{
    NumAgents = 3,
    StateSize = 24,
    ActionSize = 2,
    HiddenLayers = [256, 256],
    LearningRate = 1e-3f,
    Gamma = 0.95f
};

var system = new MADDPGSystem<float>(maddpgConfig);

// Training with centralized critic, decentralized actors
var states = multiAgentEnv.Reset();

while (!multiAgentEnv.IsDone)
{
    var actions = system.SelectActions(states);
    var (nextStates, rewards, dones) = multiAgentEnv.Step(actions);

    system.StoreTransitions(states, actions, rewards, nextStates, dones);
    system.Update();

    states = nextStates;
}
```

---

## Reward Shaping

```csharp
// Custom reward function
float ShapeReward(T[] state, int action, T[] nextState)
{
    float baseReward = env.GetReward();

    // Add shaping terms
    float progressReward = ComputeProgress(nextState);
    float safetyPenalty = ComputeSafetyViolation(nextState);

    return baseReward + 0.1f * progressReward - 0.5f * safetyPenalty;
}
```

---

## Curriculum Learning

```csharp
var curriculum = new CurriculumLearning<float>
{
    Stages = new[]
    {
        new Stage { Difficulty = 0.1f, SuccessThreshold = 0.8f },
        new Stage { Difficulty = 0.5f, SuccessThreshold = 0.7f },
        new Stage { Difficulty = 1.0f, SuccessThreshold = 0.6f }
    }
};

curriculum.OnStageComplete += (stage) =>
{
    Console.WriteLine($"Completed stage {stage}!");
};

// Training adjusts difficulty automatically
await agent.TrainWithCurriculum(env, curriculum);
```

---

## Hyperparameter Tuning

### Common Hyperparameters

| Parameter | Range | Notes |
|:----------|:------|:------|
| Learning Rate | 1e-5 to 1e-3 | Start at 3e-4 |
| Gamma | 0.9 to 0.999 | Higher for long-horizon |
| Batch Size | 32 to 512 | Larger is more stable |
| Hidden Layers | [64] to [512, 512] | Depends on complexity |

### AutoRL

```csharp
var tuner = new RLHyperparameterTuner<float>
{
    Algorithm = "PPO",
    Environment = env,
    NumTrials = 50,
    TimeoutMinutes = 60
};

var bestConfig = await tuner.OptimizeAsync();
```

---

## Saving and Loading

```csharp
// Save agent
await agent.SaveAsync("ppo_agent.aidotnet");

// Load agent
var loadedAgent = await PPOAgent<float>.LoadAsync("ppo_agent.aidotnet");

// Continue training or evaluate
```

---

## Best Practices

1. **Start with PPO**: Most stable, works on many problems
2. **Normalize observations**: Scale states to [-1, 1] or [0, 1]
3. **Reward scaling**: Keep rewards in reasonable range
4. **Frame stacking**: For partial observability (images)
5. **Parallel environments**: Speed up training with vectorized envs

---

## Debugging Tips

### Reward Not Improving?

```csharp
// Log more metrics
agent.OnUpdate += (metrics) =>
{
    Console.WriteLine($"Policy Loss: {metrics.PolicyLoss}");
    Console.WriteLine($"Value Loss: {metrics.ValueLoss}");
    Console.WriteLine($"Entropy: {metrics.Entropy}");
    Console.WriteLine($"KL Divergence: {metrics.KLDivergence}");
};
```

### Unstable Training?

- Reduce learning rate
- Increase batch size
- Add gradient clipping
- Check reward scale

---

## Next Steps

- [CartPole Sample](/samples/reinforcement-learning/CartPole/)
- [Deep Q-Learning Sample](/samples/reinforcement-learning/DeepQLearning/)
- [RL API Reference](/api/AiDotNet.ReinforcementLearning/)
