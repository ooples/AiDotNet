# Reinforcement Learning Agents

This directory contains implementations of reinforcement learning algorithms fully integrated with AiDotNet's architecture.

## Implementation Status

### ✅ Fully Implemented (Production-Ready)

1. **DQN (Deep Q-Network)** - `DQN/DQNAgent.cs`
   - Value-based, discrete actions
   - Experience replay + target network
   - Classic algorithm for Atari games
   - Status: **Complete**

2. **PPO (Proximal Policy Optimization)** - `PPO/PPOAgent.cs`
   - Policy gradient, discrete/continuous
   - Clipped objective, GAE, multi-epoch training
   - State-of-the-art, used in ChatGPT RLHF
   - Status: **Complete**

3. **SAC (Soft Actor-Critic)** - `SAC/SACAgent.cs`
   - Off-policy actor-critic, continuous
   - Maximum entropy, twin Q-networks, auto-tuning
   - Best for continuous control
   - Status: **In Progress** → Complete Next

### 📋 Critical Priority (Templates/Implementations Needed)

4. **Double DQN** - Reduces overestimation bias in Q-learning
5. **Dueling DQN** - Separates value and advantage functions
6. **Rainbow DQN** - Combines multiple DQN improvements
7. **REINFORCE** - Simplest policy gradient algorithm
8. **A2C (Advantage Actor-Critic)** - Synchronous actor-critic
9. **A3C (Asynchronous Advantage Actor-Critic)** - Parallel training version
10. **TRPO (Trust Region Policy Optimization)** - Constrained policy updates

### 🎯 High Priority

11. **DDPG (Deep Deterministic Policy Gradient)** - Deterministic continuous control
12. **TD3 (Twin Delayed DDPG)** - Improved DDPG with twin critics

### 📊 Medium Priority (Future Work)

13. **Dreamer** - Model-based, world models
14. **MuZero** - Model-based planning, AlphaGo successor
15. **World Models** - Learn dynamics model
16. **MADDPG** - Multi-agent DDPG
17. **QMIX** - Multi-agent value decomposition
18. **CQL (Conservative Q-Learning)** - Offline RL
19. **IQL (Implicit Q-Learning)** - Offline RL
20. **Decision Transformer** - Transformer-based RL
21. **Rainbow** - Combines 6 DQN extensions

## Architecture Patterns

All RL agents follow these patterns:

### Base Class Hierarchy
```csharp
ReinforcementLearningAgentBase<T>  // Base for all agents
    ↓ implements
IRLAgent<T>  // RL-specific interface
    ↓ extends
IFullModel<T, Vector<T>, Vector<T>>  // Integrates with AiDotNet
```

### Integration with AiModelBuilder

```csharp
// Training an RL agent
var agent = new DQNAgent<double>(options);

var result = await new AiModelBuilder<double, Vector<double>, Vector<double>>()
    .ConfigureEnvironment(new CartPoleEnvironment<double>())
    .ConfigureModel(agent)
    .BuildAsync(episodes: 1000);

// Using trained agent
var action = result.Predict(state);
```

### Key Components

1. **Agents** (`Agents/*/Agent.cs`)
   - Extend `ReinforcementLearningAgentBase<T>`
   - Implement `IRLAgent<T>` interface
   - Use Vector<T>, Matrix<T>, Tensor<T> exclusively

2. **Options** (`Agents/*/Options.cs`)
   - Configuration for each algorithm
   - Sensible defaults via `Default()` factory method
   - Comprehensive documentation

3. **Environments** (`Environments/*.cs`)
   - Implement `IEnvironment<T>`
   - Classic benchmarks: CartPole, MountainCar, Pendulum
   - Custom environments supported

4. **Infrastructure**
   - Replay buffers: `ReplayBuffers/`
   - Trajectories: `Common/Trajectory.cs`
   - Experience tuples: `ReplayBuffers/Experience.cs`

## Algorithm Categories

### Value-Based (Q-Learning Family)
- Learn action-value function Q(s,a)
- **Discrete actions** only
- Examples: DQN, Double DQN, Dueling DQN, Rainbow

### Policy Gradient
- Learn policy π(a|s) directly
- **Discrete or continuous** actions
- Examples: REINFORCE, PPO, TRPO

### Actor-Critic
- Learn both policy (actor) and value (critic)
- **Discrete or continuous** actions
- Examples: A2C, A3C, PPO, SAC, DDPG, TD3

### Model-Based
- Learn environment dynamics model
- Plan using the model
- Examples: Dreamer, MuZero, World Models

### Multi-Agent
- Multiple agents learning together
- Cooperative or competitive
- Examples: MADDPG, QMIX

### Offline RL
- Learn from fixed dataset (no environment interaction)
- Safe for real-world deployment
- Examples: CQL, IQL, Decision Transformer

## Type System

All implementations use AiDotNet's type system:

```csharp
Vector<T>  // States, actions, Q-values
Matrix<T>  // Parameters, gradients
Tensor<T>  // Multi-dimensional data (images, etc.)
INumericOperations<T>  // Generic numeric ops
```

**Never use:**
- `double[]`, `float[]` arrays
- `List<T>`, standard collections for numeric data
- Direct floating-point operations

## Testing

Tests are in `tests/AiDotNet.Tests/UnitTests/ReinforcementLearning/`:

```csharp
// Test agent on CartPole
[Fact]
public async Task DQNAgent_LearnCartPole()
{
    var agent = new DQNAgent<double>(options);
    var env = new CartPoleEnvironment<double>();

    var result = await new AiModelBuilder<double, Vector<double>, Vector<double>>()
        .ConfigureEnvironment(env)
        .ConfigureModel(agent)
        .BuildAsync(episodes: 500);

    // Agent should learn to balance pole
    Assert.True(TestPerformance(result, env) > 100);
}
```

## Adding New Algorithms

To add a new RL algorithm:

1. **Create Options class** (`Agents/YourAlgorithm/YourAlgorithmOptions.cs`)
   ```csharp
   public class YourAlgorithmOptions<T>
   {
       public int StateSize { get; init; }
       public int ActionSize { get; init; }
       // ... hyperparameters

       public static YourAlgorithmOptions<T> Default(...)
       {
           // Sensible defaults
       }
   }
   ```

2. **Create Agent class** (`Agents/YourAlgorithm/YourAlgorithmAgent.cs`)
   ```csharp
   public class YourAlgorithmAgent<T> : ReinforcementLearningAgentBase<T>
   {
       public override Vector<T> SelectAction(Vector<T> state, bool training = true)
       {
           // Action selection logic
       }

       public override void StoreExperience(...) {  // Experience storage  }

       public override T Train()
       {
           // Training logic
       }

       // Implement other IRLAgent<T> methods
   }
   ```

3. **Add ModelType** (in `src/Enums/ModelType.cs`)
   ```csharp
   YourAlgorithmAgent
   ```

4. **Create Tests**
   ```csharp
   public class YourAlgorithmAgentTests
   {
       [Fact]
       public async Task LearnSimpleTask() { ... }
   }
   ```

## References

- **DQN**: Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015
- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
- **SAC**: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL", 2018
- **DDPG**: Lillicrap et al., "Continuous control with deep reinforcement learning", 2015
- **TD3**: Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods", 2018

## Contributing

When implementing new algorithms:
- Follow established patterns (see DQN, PPO, SAC)
- Use Vector<T>/Matrix<T>/Tensor<T> exclusively
- Extend ReinforcementLearningAgentBase<T>
- Add comprehensive documentation
- Include unit tests
- Update this README

## License

Part of AiDotNet library - see root LICENSE file.
