# AiDotNet Reinforcement Learning Framework

## Overview

The AiDotNet Reinforcement Learning (RL) framework provides a comprehensive, extensible platform for implementing and training reinforcement learning agents. This module implements core RL concepts and algorithms following industry-standard interfaces similar to OpenAI Gym.

## Features

### Core Interfaces

- **`IEnvironment<T>`**: Standard interface for RL environments (Gym-like API)
- **`IReplayBuffer<T>`**: Interface for experience replay buffers
- **`IRLAgent<T>`**: Interface for RL agents
- **`IPolicy<T>`**: Interface for exploration/exploitation policies

### Implemented Components

#### Replay Buffers
- **UniformReplayBuffer**: Standard experience replay with uniform sampling
- **PrioritizedReplayBuffer**: Priority-based experience replay (PER) for more efficient learning

#### Policies
- **EpsilonGreedyPolicy**: ε-greedy exploration strategy
- **SoftmaxPolicy**: Boltzmann exploration with temperature parameter
- **GreedyPolicy**: Pure exploitation (no exploration)

#### Agents
- **DQNAgent**: Complete Deep Q-Network implementation with:
  - Experience replay
  - Target network
  - Configurable exploration
  - Flexible policy selection

#### Environments
- **CartPoleEnvironment**: Classic CartPole balancing task for testing and learning

## Quick Start

### Creating a DQN Agent

```csharp
using AiDotNet.ReinforcementLearning;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.ReinforcementLearning.Environments;
using AiDotNet.ReinforcementLearning.Policies;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.NeuralNetworks;
using AiDotNet.Models;

// Create environment
var env = new CartPoleEnvironment<double>(maxSteps: 500);

// Create neural network architecture
var architecture = new NeuralNetworkArchitecture<double>
{
    InputSize = env.ObservationSpaceDimension, // 4 for CartPole
    OutputSize = env.ActionSpaceSize,           // 2 for CartPole
    HiddenLayers = new[] { 64, 64 },
    TaskType = NeuralNetworkTaskType.ReinforcementLearning
};

// Create Q-network and target network
var qNetwork = new DeepQNetwork<double>(architecture);
var targetNetwork = new DeepQNetwork<double>(architecture);

// Create replay buffer
var replayBuffer = new UniformReplayBuffer<double>(capacity: 100000);

// Create exploration policy
var policy = new EpsilonGreedyPolicy<double>(
    actionSpaceSize: env.ActionSpaceSize,
    epsilonStart: 1.0,
    epsilonMin: 0.01,
    epsilonDecay: 0.995
);

// Create DQN agent
var agent = new DQNAgent<double>(
    qNetwork: qNetwork,
    targetNetwork: targetNetwork,
    replayBuffer: replayBuffer,
    policy: policy,
    batchSize: 32,
    gamma: 0.99,
    learningRate: 0.001,
    targetUpdateFrequency: 100
);
```

### Training Loop

```csharp
int numEpisodes = 500;
int maxStepsPerEpisode = 500;

for (int episode = 0; episode < numEpisodes; episode++)
{
    var state = env.Reset();
    double episodeReward = 0;
    bool done = false;
    int steps = 0;

    while (!done && steps < maxStepsPerEpisode)
    {
        // Select action
        int action = agent.SelectAction(state, training: true);

        // Take action in environment
        var (nextState, reward, isDone, info) = env.Step(action);

        // Store experience
        agent.StoreExperience(state, action, reward, nextState, isDone);

        // Train agent (if enough experiences)
        double loss = agent.Train();

        // Update state
        state = nextState;
        done = isDone;
        episodeReward += reward;
        steps++;
    }

    Console.WriteLine($"Episode {episode + 1}: Reward = {episodeReward}, Steps = {steps}, Epsilon = {policy.Epsilon:F3}");

    // Save checkpoint every 100 episodes
    if ((episode + 1) % 100 == 0)
    {
        agent.Save($"dqn_checkpoint_episode_{episode + 1}.bin");
    }
}
```

### Evaluation

```csharp
// Load trained agent
agent.Load("dqn_checkpoint_episode_500.bin");

// Evaluate for 10 episodes
int evalEpisodes = 10;
double totalReward = 0;

for (int episode = 0; episode < evalEpisodes; episode++)
{
    var state = env.Reset();
    double episodeReward = 0;
    bool done = false;

    while (!done)
    {
        // Select best action (no exploration)
        int action = agent.SelectAction(state, training: false);

        // Take action
        var (nextState, reward, isDone, info) = env.Step(action);

        state = nextState;
        done = isDone;
        episodeReward += reward;

        // Optional: Render environment
        env.Render();
    }

    totalReward += episodeReward;
    Console.WriteLine($"Eval Episode {episode + 1}: Reward = {episodeReward}");
}

Console.WriteLine($"Average Evaluation Reward: {totalReward / evalEpisodes:F2}");
```

## Architecture

### Directory Structure

```
src/ReinforcementLearning/
├── Interfaces/
│   ├── IEnvironment.cs          # Environment interface
│   ├── IReplayBuffer.cs         # Replay buffer interface
│   ├── IRLAgent.cs              # RL agent interface
│   └── IPolicy.cs               # Policy interface
├── Agents/
│   └── DQNAgent.cs              # DQN agent implementation
├── Environments/
│   └── CartPoleEnvironment.cs   # Example CartPole environment
├── ReplayBuffers/
│   ├── UniformReplayBuffer.cs   # Uniform sampling buffer
│   └── PrioritizedReplayBuffer.cs # Priority-based buffer
├── Policies/
│   ├── EpsilonGreedyPolicy.cs   # ε-greedy policy
│   ├── SoftmaxPolicy.cs         # Softmax/Boltzmann policy
│   └── GreedyPolicy.cs          # Greedy policy
├── Enums/
│   ├── RLAlgorithmType.cs       # RL algorithm types
│   └── ExplorationStrategyType.cs # Exploration strategy types
└── README.md                    # This file
```

## Algorithms

### Implemented
- **Deep Q-Network (DQN)**: Value-based method for discrete action spaces
  - Experience replay for sample efficiency
  - Target network for stability
  - Flexible exploration strategies

### Planned (Future Releases)
- **Double DQN**: Reduces overestimation bias
- **Dueling DQN**: Separates value and advantage streams
- **Rainbow DQN**: Combines multiple improvements
- **REINFORCE**: Policy gradient method
- **A2C/A3C**: Advantage Actor-Critic variants
- **PPO**: Proximal Policy Optimization
- **DDPG/TD3**: Continuous control actor-critic methods
- **SAC**: Soft Actor-Critic with maximum entropy
- **Model-based methods**: World models, planning

## Exploration Strategies

### ε-Greedy (EpsilonGreedyPolicy)
- Simple and effective
- Balances random exploration and exploitation
- Supports epsilon decay schedules
- **Use when**: Standard approach for most problems

### Softmax/Boltzmann (SoftmaxPolicy)
- Probability-based exploration
- Favors high-value actions while still exploring
- Temperature-based control
- **Use when**: You want smooth, value-proportional exploration

### Greedy (GreedyPolicy)
- No exploration, pure exploitation
- Always selects best action
- **Use when**: Evaluation/deployment after training

## Replay Buffers

### Uniform Replay Buffer
- Standard experience replay
- Uniform random sampling
- Simple and effective
- **Use when**: Standard DQN training

### Prioritized Experience Replay (PER)
- Samples important experiences more frequently
- Based on TD error magnitude
- More sample-efficient
- **Use when**: Sparse rewards or sample efficiency is critical

## Creating Custom Components

### Custom Environment

```csharp
public class MyEnvironment<T> : IEnvironment<T>
{
    public int ObservationSpaceDimension => 4;
    public int ActionSpaceSize => 2;

    public Tensor<T> Reset()
    {
        // Reset environment to initial state
        // Return initial observation
    }

    public (Tensor<T> nextState, T reward, bool done, Dictionary<string, object>? info) Step(int action)
    {
        // Execute action
        // Update environment state
        // Return (next_state, reward, done, info)
    }

    public void Render() { /* Optional visualization */ }
    public void Close() { /* Cleanup resources */ }
    public void Seed(int seed) { /* Set random seed */ }
}
```

### Custom Policy

```csharp
public class MyPolicy<T> : IPolicy<T>
{
    public int SelectAction(Tensor<T> state, Tensor<T>? qValues = null)
    {
        // Implement action selection logic
    }

    public Tensor<T> GetActionProbabilities(Tensor<T> state, Tensor<T>? qValues = null)
    {
        // Return action probability distribution
    }

    public void Update()
    {
        // Update policy parameters (e.g., decay exploration)
    }
}
```

## Best Practices

### Hyperparameter Tuning

1. **Learning Rate**: Start with 0.001 or 0.0001
2. **Batch Size**: 32-64 for small problems, 128-256 for larger
3. **Replay Buffer Size**: 10k-100k for simple, 1M+ for complex (Atari)
4. **Discount Factor (γ)**: 0.99 is standard, 0.95-0.999 range
5. **Target Update Frequency**: 100-1000 steps
6. **Epsilon Schedule**: Start 1.0, decay to 0.01-0.1 over training

### Training Tips

1. **Warm-up**: Collect random experiences before training (1000-10000 steps)
2. **Monitoring**: Track episode rewards, loss, epsilon, and training steps
3. **Checkpoints**: Save model regularly during training
4. **Evaluation**: Periodically evaluate with exploration disabled
5. **Hyperparameter Search**: Try different values systematically
6. **Seeding**: Use seeds for reproducibility during development

### Common Issues

**Training is unstable:**
- Reduce learning rate
- Increase target network update frequency
- Use smaller batch size
- Check reward scaling

**Agent not exploring:**
- Verify epsilon is decaying properly
- Try softmax policy instead
- Increase initial epsilon

**Learning is too slow:**
- Increase learning rate (carefully)
- Use prioritized replay
- Check network architecture
- Verify reward signal is informative

**Memory issues:**
- Reduce replay buffer capacity
- Reduce batch size
- Use simpler network architecture

## Examples

See the `examples/` directory for complete working examples:
- Basic DQN on CartPole
- Custom environment implementation
- Advanced training with logging and visualization

## Performance Benchmarks

### CartPole-v1 (500 step limit)
- **Solve criteria**: Average reward ≥ 475 over 100 consecutive episodes
- **Expected time to solve**: 200-300 episodes with default hyperparameters
- **Training time**: ~5-10 minutes on CPU

## References

### Papers
- [Playing Atari with Deep Reinforcement Learning (DQN)](https://arxiv.org/abs/1312.5602) - Mnih et al., 2013
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) - Mnih et al., 2015
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) - van Hasselt et al., 2015
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) - Wang et al., 2015
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) - Schaul et al., 2015

### Resources
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Sutton & Barto: Reinforcement Learning Book](http://www.incompleteideas.net/book/the-book.html)

## Contributing

Contributions are welcome! Areas for contribution:
- Additional RL algorithms (A2C, PPO, SAC, etc.)
- More environment implementations
- Improved visualization tools
- Performance optimizations
- Documentation improvements

## License

This project is part of AiDotNet and follows the same license.
