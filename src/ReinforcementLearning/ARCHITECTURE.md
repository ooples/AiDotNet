# AiDotNet Reinforcement Learning Architecture

## Overview

This document describes the comprehensive architecture for integrating Reinforcement Learning into AiDotNet, following the established patterns of `PredictionModelBuilder`, `OptimizerBase`, and `NeuralNetworkBase`.

## Core Principles

### 1. Facade Pattern Integration
- Users interact through `RLModelBuilder<T>` (similar to `PredictionModelBuilder<T>`)
- All complexity hidden behind simple builder methods
- Sensible industry-standard defaults for all parameters
- No RL knowledge required for basic usage

### 2. Type System Integration
- All data uses `Vector<T>`, `Matrix<T>`, `Tensor<T>` (never raw arrays)
- Implements `IFullModel<T, Tensor<T>, Vector<T>>` for consistency
- Uses `INumericOperations<T>` for all numeric operations
- Generic type parameter `T` throughout (float/double support)

### 3. Base Class Hierarchy
```
OptimizerBase<T, TInput, TOutput>
    └─ ReinforcementLearningAgentBase<T>
        ├─ ValueBasedAgentBase<T>
        │   ├─ DQNAgent<T>
        │   ├─ DoubleDQNAgent<T>
        │   ├─ DuelingDQNAgent<T>
        │   └─ RainbowDQNAgent<T>
        ├─ PolicyGradientAgentBase<T>
        │   ├─ REINFORCEAgent<T>
        │   ├─ A2CAgent<T>
        │   ├─ A3CAgent<T>
        │   ├─ PPOAgent<T>
        │   └─ TRPOAgent<T>
        ├─ ActorCriticAgentBase<T>
        │   ├─ DDPGAgent<T>
        │   ├─ TD3Agent<T>
        │   └─ SACAgent<T>
        ├─ ModelBasedAgentBase<T>
        │   ├─ DreamerAgent<T>
        │   ├─ MuZeroAgent<T>
        │   └─ WorldModelsAgent<T>
        ├─ MultiAgentBase<T>
        │   ├─ MADDPGAgent<T>
        │   └─ QMIXAgent<T>
        └─ OfflineRLAgentBase<T>
            ├─ CQLAgent<T>
            ├─ IQLAgent<T>
            └─ DecisionTransformerAgent<T>
```

## User-Facing API

### Simple Usage (Beginner)
```csharp
// Absolute minimum - library chooses everything
var model = new RLModelBuilder<double>()
    .WithEnvironment(new CartPoleEnvironment<double>())
    .Build();

var result = model.Train(episodes: 500);

// Use trained model
var action = result.GetAction(state);
```

### Intermediate Usage
```csharp
var model = new RLModelBuilder<double>()
    .WithEnvironment(env)
    .WithAlgorithm(RLAlgorithm.PPO)  // Choose algorithm
    .WithNetworkArchitecture(arch)    // Custom network
    .Build();
```

### Advanced Usage (Expert)
```csharp
var model = new RLModelBuilder<double>()
    .WithEnvironment(env)
    .WithAlgorithm(RLAlgorithm.SAC)
    .WithNetworkArchitecture(policyArch, valueArch)
    .ConfigureLearning(options =>
    {
        options.LearningRate = 0.0003;
        options.DiscountFactor = 0.99;
        options.BatchSize = 256;
    })
    .ConfigureExploration(options =>
    {
        options.Strategy = ExplorationStrategy.EntropyRegularization;
        options.EntropyCoefficient = 0.2;
    })
    .ConfigureReplayBuffer(options =>
    {
        options.Type = ReplayBufferType.Prioritized;
        options.Capacity = 1000000;
        options.Alpha = 0.6;
        options.Beta = 0.4;
    })
    .WithDistributedTraining(DistributedStrategy.DDP)
    .WithMixedPrecision()
    .Build();
```

## RLModelResult<T> (Return Type)

Mirrors `PredictionModelResult<T>`:

```csharp
public class RLModelResult<T> : IFullModel<T, Tensor<T>, Vector<T>>
{
    // Trained agent
    internal ReinforcementLearningAgentBase<T> Agent { get; }

    // Training history
    internal RLTrainingResult<T> TrainingResult { get; }

    // Environment info
    internal EnvironmentMetadata Metadata { get; }

    // Main usage methods
    public Vector<T> GetAction(Tensor<T> state);
    public Vector<T> Predict(Tensor<T> state); // IFullModel interface
    public void ContinueTraining(int additionalEpisodes);
    public void Adapt(IEnvironment<T> newEnv, int episodes);

    // Serialization
    public void SaveModel(string filepath);
    public static RLModelResult<T> LoadModel(string filepath);

    // Analysis
    public Dictionary<string, T> GetMetrics();
    public List<T> GetEpisodeRewards();
    public List<T> GetLossHistory();
}
```

## Environment Interface

```csharp
public interface IEnvironment<T>
{
    // Gym-compatible interface
    int ObservationSpaceDimension { get; }
    int ActionSpaceDimension { get; }
    bool IsDiscrete { get; }

    Vector<T> Reset();
    (Vector<T> nextState, T reward, bool done, Dictionary<string, object> info) Step(Vector<T> action);
    void Render();
    void Close();
    void Seed(int seed);
}

public abstract class EnvironmentBase<T> : IEnvironment<T>
{
    protected readonly INumericOperations<T> NumOps;
    protected readonly Random Random;

    // Implements common functionality
}
```

## Algorithm Specifications

### CRITICAL Priority - Value-Based Methods

#### 1. DQN (Deep Q-Network)
**Default Hyperparameters:**
- Learning Rate: 0.00025
- Discount Factor (γ): 0.99
- Replay Buffer Size: 100,000
- Batch Size: 32
- Target Update Frequency: 1,000 steps
- Epsilon Start: 1.0
- Epsilon End: 0.01
- Epsilon Decay: 0.995
- Network: [64, 64] hidden layers

**Components:**
- Q-Network: `NeuralNetwork<T>` (existing)
- Target Network: Periodic copy of Q-Network
- Replay Buffer: `UniformReplayBuffer<T>`
- Policy: `EpsilonGreedyPolicy<T>`

#### 2. Double DQN
**Inherits DQN defaults, adds:**
- Uses Q-Network for action selection
- Uses Target Network for value evaluation
- Reduces overestimation bias

#### 3. Dueling DQN
**Inherits DQN defaults, modifies:**
- Network Architecture: Splits into Value and Advantage streams
- Value Stream: [64] → Scalar
- Advantage Stream: [64] → Action dimensions
- Combines: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))

#### 4. Rainbow DQN
**Combines:**
- Double Q-learning
- Prioritized Replay (α=0.6, β=0.4→1.0)
- Dueling Networks
- Multi-step Learning (n=3)
- Distributional RL (51 atoms, V_min=-10, V_max=10)
- Noisy Networks (σ=0.5)

### CRITICAL Priority - Policy Gradient Methods

#### 5. REINFORCE
**Default Hyperparameters:**
- Learning Rate: 0.001
- Discount Factor (γ): 0.99
- Baseline: None (vanilla) or Value function
- Network: [64, 64] policy network

**Components:**
- Policy Network: `NeuralNetwork<T>` → Action probabilities
- No replay buffer (on-policy)
- Monte Carlo returns

#### 6. A2C (Advantage Actor-Critic)
**Default Hyperparameters:**
- Learning Rate: 0.0007
- Discount Factor (γ): 0.99
- Entropy Coefficient: 0.01
- Value Loss Coefficient: 0.5
- Max Gradient Norm: 0.5
- N-step: 5
- Policy Network: [64, 64]
- Value Network: [64, 64]

**Components:**
- Actor Network: Policy
- Critic Network: Value function
- Advantage = TD-error

#### 7. A3C (Asynchronous A2C)
**Inherits A2C defaults, adds:**
- Multiple parallel workers (default: 4)
- Asynchronous parameter updates
- Shared global parameters

#### 8. PPO (Proximal Policy Optimization)
**Default Hyperparameters:**
- Learning Rate: 0.0003
- Discount Factor (γ): 0.99
- GAE Lambda (λ): 0.95
- Clip Epsilon: 0.2
- Epochs per update: 10
- Batch Size: 64
- Minibatches: 4
- Entropy Coefficient: 0.01
- Value Loss Coefficient: 0.5
- Max Gradient Norm: 0.5
- Policy Network: [64, 64]
- Value Network: [64, 64]

**Components:**
- Actor Network: Policy
- Critic Network: Value function
- Clipped surrogate objective
- GAE for advantage estimation

#### 9. TRPO (Trust Region Policy Optimization)
**Default Hyperparameters:**
- Max KL Divergence: 0.01
- Discount Factor (γ): 0.99
- GAE Lambda (λ): 0.95
- Damping: 0.1
- CG Iterations: 10
- Backtrack Iterations: 10
- Policy Network: [64, 64]
- Value Network: [64, 64]

**Components:**
- Actor Network: Policy
- Critic Network: Value function
- Conjugate Gradient for natural gradient
- Line search for KL constraint

### HIGH Priority - Actor-Critic Methods

#### 10. DDPG (Deep Deterministic Policy Gradient)
**Default Hyperparameters:**
- Learning Rate Actor: 0.0001
- Learning Rate Critic: 0.001
- Discount Factor (γ): 0.99
- Tau (soft update): 0.005
- Replay Buffer Size: 1,000,000
- Batch Size: 128
- Exploration Noise: Ornstein-Uhlenbeck (θ=0.15, σ=0.2)
- Actor Network: [400, 300]
- Critic Network: [400, 300]

**Components:**
- Actor Network: Deterministic policy
- Critic Network: Q-function
- Target Actor & Critic Networks
- Replay Buffer
- OU Noise for exploration

#### 11. TD3 (Twin Delayed DDPG)
**Inherits DDPG defaults, adds:**
- Twin Critics (2 Q-networks, use minimum)
- Delayed Policy Updates (update actor every 2 critic updates)
- Target Policy Smoothing (add noise to target actions, σ=0.2, clip=0.5)
- Learning Rate: 0.001 (both actor and critic)

#### 12. SAC (Soft Actor-Critic)
**Default Hyperparameters:**
- Learning Rate: 0.0003 (all networks)
- Discount Factor (γ): 0.99
- Tau (soft update): 0.005
- Replay Buffer Size: 1,000,000
- Batch Size: 256
- Initial Temperature (α): 0.2 (auto-tuned)
- Target Entropy: -dim(action_space)
- Actor Network: [256, 256]
- Critic Networks: 2× [256, 256]

**Components:**
- Actor Network: Stochastic policy (Gaussian)
- Twin Critic Networks: Q-functions
- Temperature Parameter: Entropy regularization
- Automatic entropy tuning

### MEDIUM Priority - Model-Based RL

#### 13. Dreamer
**Default Hyperparameters:**
- Learning Rate: 0.0001
- Discount Factor (γ): 0.99
- Imagination Horizon: 15 steps
- Model Rollouts per step: 5
- Free Bits: 1.0
- KL Scale: 1.0
- Model Network: RSSM [200 stochastic, 200 deterministic]
- Actor/Critic: [400, 400, 400, 400]

**Components:**
- World Model: Recurrent State Space Model (RSSM)
  - Representation: Encoder
  - Transition: Dynamics model
  - Observation: Decoder
  - Reward: Predictor
- Actor: Learns in imagination
- Critic: Value function in imagination

#### 14. MuZero
**Default Hyperparameters:**
- Learning Rate: 0.0002
- Discount Factor (γ): 0.997
- MCTS Simulations: 800
- Temperature: 1.0 (exploration)
- Dirichlet Alpha: 0.3
- Exploration Fraction: 0.25
- Network: ResNet [256 channels]
- Value Support Size: 601

**Components:**
- Representation Network: State encoding
- Dynamics Network: Next state prediction
- Prediction Network: Policy and value
- MCTS: Planning with learned model

#### 15. World Models
**Default Hyperparameters:**
- VAE Latent Dimension: 32
- RNN Hidden Size: 256
- Temperature: 1.0
- Controller Network: Linear [32+256 → action]

**Components:**
- Vision Module (V): VAE for encoding observations
- Memory Module (M): RNN for temporal dynamics
- Controller (C): Simple linear policy
- Train in imagination using M

### MEDIUM Priority - Multi-Agent RL

#### 16. MADDPG (Multi-Agent DDPG)
**Default Hyperparameters:**
- Inherits DDPG defaults per agent
- Centralized Critic: Sees all observations/actions
- Decentralized Actors: See only own observation

**Components:**
- Per-Agent Actor: Local observations → Actions
- Centralized Critic: Global state → Q-values
- Experience replay shared or separate

#### 17. QMIX
**Default Hyperparameters:**
- Learning Rate: 0.0005
- Discount Factor (γ): 0.99
- Mixing Network: Hypernetwork [64, 64]
- Agent Networks: [64] per agent
- Epsilon Start: 1.0
- Epsilon End: 0.05
- Epsilon Decay: 0.995

**Components:**
- Agent Networks: Individual Q-networks
- Mixing Network: Combines Q-values monotonically
- Hypernetwork: Generates mixing network weights from state

### MEDIUM Priority - Offline RL

#### 18. CQL (Conservative Q-Learning)
**Default Hyperparameters:**
- Inherits SAC defaults
- CQL Alpha: 1.0 (conservatism level)
- CQL Temperature: 1.0
- Lagrange Threshold: 0.0 (optional constraint)

**Components:**
- Inherits SAC architecture
- Conservative Q-function regularization
- Fixed dataset (no environment interaction)

#### 19. IQL (Implicit Q-Learning)
**Default Hyperparameters:**
- Learning Rate: 0.0003
- Discount Factor (γ): 0.99
- Tau (soft update): 0.005
- Expectile (τ): 0.7
- Temperature (β): 3.0
- Networks: [256, 256] each

**Components:**
- Value Network: Expectile regression on Q
- Q-Networks: Twin critics
- Actor: Advantage-weighted regression
- No OOD actions (fully offline)

#### 20. Decision Transformer
**Default Hyperparameters:**
- Learning Rate: 0.0001
- Context Length: 20 timesteps
- Transformer Layers: 3
- Attention Heads: 1
- Embedding Dimension: 128
- Dropout: 0.1

**Components:**
- Transformer Model: Processes (R, s, a) sequences
- Autoregressive prediction of actions
- Conditioned on desired return-to-go
- No value function or policy gradient

## Integration with PredictionModelBuilder Pattern

### RLModelBuilder<T> Implementation

```csharp
public class RLModelBuilder<T>
{
    private IEnvironment<T>? _environment;
    private RLAlgorithm _algorithm = RLAlgorithm.PPO; // Best default
    private NeuralNetworkArchitecture<T>? _policyArchitecture;
    private NeuralNetworkArchitecture<T>? _valueArchitecture;
    private RLLearningOptions<T>? _learningOptions;
    private RLExplorationOptions? _explorationOptions;
    private RLReplayBufferOptions? _replayOptions;
    private int? _seed;

    public RLModelBuilder<T> WithEnvironment(IEnvironment<T> environment);
    public RLModelBuilder<T> WithAlgorithm(RLAlgorithm algorithm);
    public RLModelBuilder<T> WithNetworkArchitecture(NeuralNetworkArchitecture<T> policy);
    public RLModelBuilder<T> WithNetworkArchitecture(NeuralNetworkArchitecture<T> policy, NeuralNetworkArchitecture<T> value);
    public RLModelBuilder<T> ConfigureLearning(Action<RLLearningOptions<T>> configure);
    public RLModelBuilder<T> ConfigureExploration(Action<RLExplorationOptions> configure);
    public RLModelBuilder<T> ConfigureReplayBuffer(Action<RLReplayBufferOptions> configure);
    public RLModelBuilder<T> WithSeed(int seed);

    public RLModelResult<T> Build()
    {
        // 1. Validate required components
        if (_environment == null)
            throw new InvalidOperationException("Environment is required");

        // 2. Create agent based on algorithm
        var agent = CreateAgent();

        // 3. Return result wrapper
        return new RLModelResult<T>(agent, _environment);
    }

    public RLModelResult<T> BuildAndTrain(int episodes)
    {
        var result = Build();
        result.Train(episodes);
        return result;
    }
}
```

### Default Network Architectures

The builder automatically creates sensible architectures based on environment:

```csharp
private NeuralNetworkArchitecture<T> CreateDefaultArchitecture(IEnvironment<T> env, NetworkType type)
{
    var inputSize = env.ObservationSpaceDimension;
    var outputSize = env.ActionSpaceDimension;

    return new NeuralNetworkArchitecture<T>
    {
        InputSize = inputSize,
        OutputSize = outputSize,
        HiddenLayers = type switch
        {
            NetworkType.Policy => [64, 64],           // Small envs
            NetworkType.Value => [64, 64],            // Small envs
            NetworkType.LargePolicy => [256, 256],    // Pixel envs
            NetworkType.LargeValue => [256, 256],     // Pixel envs
            _ => [64, 64]
        },
        ActivationFunction = ActivationFunction.ReLU,
        OutputActivation = type == NetworkType.Policy
            ? ActivationFunction.Tanh  // Continuous actions
            : ActivationFunction.None  // Values
    };
}
```

## Storage and Serialization

### File Format
```
model.rl
├─ metadata.json           # Algorithm, hyperparameters, environment info
├─ policy_network.bin      # Policy network weights
├─ value_network.bin       # Value network weights (if applicable)
├─ target_networks.bin     # Target networks (if applicable)
├─ replay_buffer.bin       # Replay buffer (optional)
└─ training_history.json   # Episode rewards, losses, metrics
```

### Save/Load
```csharp
// Save
result.SaveModel("trained_agent.rl");

// Load
var loaded = RLModelResult<double>.LoadModel("trained_agent.rl");
var action = loaded.GetAction(state);
```

## Environment Library

### Included Environments

1. **Classic Control**
   - CartPole
   - Pendulum
   - MountainCar
   - Acrobot

2. **Continuous Control**
   - Reacher
   - Pusher
   - HalfCheetah (simplified)

3. **Multi-Agent**
   - SimpleSpread
   - CooperativeNavigation
   - Predator-Prey

### Custom Environment Template

```csharp
public class MyEnvironment<T> : EnvironmentBase<T>
{
    public override int ObservationSpaceDimension => 4;
    public override int ActionSpaceDimension => 2;
    public override bool IsDiscrete => false;

    public override Vector<T> Reset()
    {
        // Reset environment state
        // Return initial observation
    }

    public override (Vector<T>, T, bool, Dictionary<string, object>) Step(Vector<T> action)
    {
        // Execute action
        // Update state
        // Calculate reward
        // Check if done
        // Return (next_state, reward, done, info)
    }
}
```

## Testing Strategy

### Unit Tests
- Each algorithm: Basic training test
- Each component: Replay buffer, policies, etc.
- Environment: Reset, step, termination

### Integration Tests
- Full training pipeline
- Save/load models
- Transfer learning

### Benchmark Tests
- CartPole: Solve in < 200 episodes
- Pendulum: Score > -200 in 100 episodes
- Ensure performance matches literature

## Performance Considerations

### Memory Usage
- Replay buffers are largest component (configurable size)
- Use mixed precision for GPU training (FP16/FP32)
- Lazy loading for model-based algorithms

### Compute Optimization
- Vectorized environments (parallel rollouts)
- GPU acceleration for network updates
- Distributed training (A3C, IMPALA patterns)

### Monitoring
- Episode reward curves
- Loss curves
- Q-value distributions
- Policy entropy
- Gradient norms

## Migration Path

### Phase 1: Core Infrastructure (Week 1)
- ReinforcementLearningAgentBase
- RLModelBuilder
- RLModelResult
- IEnvironment interface
- Basic environments

### Phase 2: Value-Based (Week 2)
- DQN, Double DQN, Dueling DQN, Rainbow
- Replay buffers
- Policies

### Phase 3: Policy Gradient (Week 3)
- REINFORCE, A2C, A3C
- PPO, TRPO

### Phase 4: Actor-Critic (Week 4)
- DDPG, TD3, SAC

### Phase 5: Advanced (Week 5-6)
- Model-based: Dreamer, MuZero, World Models
- Multi-agent: MADDPG, QMIX
- Offline: CQL, IQL, Decision Transformer

## Open Questions

1. **Multi-task RL**: Support for multiple environments in one model?
2. **Curriculum Learning**: Automatic difficulty progression?
3. **Meta-RL**: Integration with existing MetaLearning module?
4. **Imitation Learning**: Behavioral cloning, GAIL, etc.?
5. **Hierarchical RL**: Options framework, Feudal networks?

## References

- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [RLlib Documentation](https://docs.ray.io/en/latest/rllib/index.html)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
