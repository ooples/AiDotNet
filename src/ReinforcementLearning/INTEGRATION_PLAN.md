# Reinforcement Learning Integration into AiDotNet

## Core Principle: Single Entry Point

**All ML/AI functionality goes through `PredictionModelBuilder<T, TInput, TOutput>`**

## RL Integration Pattern

### How RL Fits the Existing Architecture

1. **RL Agents ARE Models**
   - Implement `IFullModel<T, Tensor<T>, Vector<T>>`
   - Input: State observations (Tensor<T>)
   - Output: Actions (Vector<T>)

2. **RL Training IS Optimization**
   - Agents are trained through environment interaction
   - Uses existing optimizer infrastructure where applicable
   - Follows `BuildAsync()` pattern (like meta-learning)

3. **Returns Standard Result**
   - Returns `PredictionModelResult<T, Tensor<T>, Vector<T>>`
   - Works with existing serialization, metrics, etc.

## User API

### Simple Usage
```csharp
var result = await new PredictionModelBuilder<double, Tensor<double>, Vector<double>>()
    .ConfigureEnvironment(new CartPoleEnvironment<double>())
    .ConfigureModel(new DQNAgent<double>())  // RL agent is just another model type
    .BuildAsync();  // Trains agent in environment, no x/y needed

// Use trained agent
var action = result.Predict(stateObservation);
```

### With Configuration
```csharp
var agent = new PPOAgent<double>(new PPOOptions<double>
{
    LearningRate = NumOps.FromDouble(0.0003),
    ClipEpsilon = 0.2,
    EntropyCoefficient = 0.01
});

var result = await new PredictionModelBuilder<double, Tensor<double>, Vector<double>>()
    .ConfigureEnvironment(env)
    .ConfigureModel(agent)
    .ConfigureOptimizer(optimizer)  // Optional: custom optimizer for policy updates
    .ConfigureMixedPrecision()      // Works with RL too!
    .BuildAsync(episodes: 1000);    // Overload for RL training
```

## Extension to PredictionModelBuilder

### New Methods (minimal additions)

```csharp
public class PredictionModelBuilder<T, TInput, TOutput>
{
    private IEnvironment<T>? _environment;
    private int _trainingEpisodes = 1000;

    /// <summary>
    /// Configures the environment for reinforcement learning.
    /// </summary>
    public IPredictionModelBuilder<T, TInput, TOutput> ConfigureEnvironment(IEnvironment<T> environment)
    {
        _environment = environment;
        return this;
    }

    /// <summary>
    /// Builds and trains an RL agent in the configured environment.
    /// </summary>
    public async Task<PredictionModelResult<T, TInput, TOutput>> BuildAsync(int episodes)
    {
        if (_environment == null)
            throw new InvalidOperationException("Environment required for RL training");

        if (_model == null || _model is not IRLAgent<T>)
            throw new InvalidOperationException("RL agent required (use ConfigureModel with DQNAgent, PPOAgent, etc.)");

        var agent = (IRLAgent<T>)_model;

        // Train agent in environment
        for (int episode = 0; episode < episodes; episode++)
        {
            var state = _environment.Reset();
            bool done = false;

            while (!done)
            {
                var action = agent.SelectAction(state);
                var (nextState, reward, isDone, _) = _environment.Step(action);
                agent.StoreExperience(state, action, reward, nextState, isDone);
                agent.Train();
                state = nextState;
                done = isDone;
            }
        }

        // Return standard result
        return new PredictionModelResult<T, TInput, TOutput>(
            model: _model,
            optimizationResult: CreateRLOptimizationResult(agent),
            normalizationInfo: new NormalizationInfo<T, TInput, TOutput>() // RL doesn't normalize like supervised
        );
    }
}
```

## RL Agent Interface

```csharp
/// <summary>
/// Marker interface for RL agents that integrate with PredictionModelBuilder.
/// </summary>
public interface IRLAgent<T> : IFullModel<T, Tensor<T>, Vector<T>>
{
    /// <summary>
    /// Selects an action given current state.
    /// </summary>
    Vector<T> SelectAction(Tensor<T> state);

    /// <summary>
    /// Stores experience for training.
    /// </summary>
    void StoreExperience(Tensor<T> state, Vector<T> action, T reward, Tensor<T> nextState, bool done);

    /// <summary>
    /// Performs one training step.
    /// </summary>
    void Train();

    /// <summary>
    /// Gets training metrics.
    /// </summary>
    Dictionary<string, T> GetMetrics();
}
```

## Agent Implementations

Each agent implements both `IRLAgent<T>` AND `IFullModel<T, Tensor<T>, Vector<T>>`:

```csharp
public class DQNAgent<T> : IRLAgent<T>, IFullModel<T, Tensor<T>, Vector<T>>
{
    private readonly DeepQNetwork<T> _qNetwork;  // Uses existing NeuralNetwork
    private readonly DeepQNetwork<T> _targetNetwork;
    private readonly IReplayBuffer<T> _replayBuffer;
    private readonly DQNOptions<T> _options;

    // IFullModel implementation
    public Vector<T> Predict(Tensor<T> state) => SelectAction(state);
    public ModelMetadata<T> GetModelMetadata() => /* ... */;
    public void Train(Tensor<T> input, Vector<T> target) => throw new NotSupportedException("Use RL training");

    // IRLAgent implementation
    public Vector<T> SelectAction(Tensor<T> state) { /* epsilon-greedy */ }
    public void StoreExperience(...) { /* add to replay buffer */ }
    public void Train() { /* DQN update logic */ }

    // Serialization
    public byte[] Serialize() { /* ... */ }
    public void Deserialize(byte[] data) { /* ... */ }

    // Other IFullModel methods...
}
```

## Factory Integration

```csharp
// In OptimizerFactory or ModelFactory
public static class RLAgentFactory
{
    public static IRLAgent<T> CreateAgent<T>(
        RLAlgorithmType algorithm,
        NeuralNetworkArchitecture<T> architecture,
        RLOptions<T> options)
    {
        return algorithm switch
        {
            RLAlgorithmType.DQN => new DQNAgent<T>(architecture, options.ToDQNOptions()),
            RLAlgorithmType.PPO => new PPOAgent<T>(architecture, options.ToPPOOptions()),
            RLAlgorithmType.SAC => new SACAgent<T>(architecture, options.ToSACOptions()),
            // ... all 21 algorithms
            _ => throw new ArgumentException($"Unknown algorithm: {algorithm}")
        };
    }
}
```

## Enum Integration

Add to existing enums:

```csharp
// In ModelType enum
public enum ModelType
{
    // ... existing types
    DQNAgent,
    DoubleDQNAgent,
    DuelingDQNAgent,
    RainbowDQNAgent,
    REINFORCEAgent,
    A2CAgent,
    PPOAgent,
    // ... all RL types
}
```

## Implementation Checklist

### Phase 1: Infrastructure (Week 1)
- [ ] `IEnvironment<T>` interface
- [ ] `IRLAgent<T>` interface
- [ ] `ConfigureEnvironment()` in PredictionModelBuilder
- [ ] `BuildAsync(int episodes)` overload
- [ ] Basic environment implementations (CartPole, etc.)
- [ ] Replay buffer implementations

### Phase 2: Value-Based Algorithms (Week 2)
- [ ] `DQNAgent<T>` - Complete implementation
- [ ] `DoubleDQNAgent<T>` - Extends DQN
- [ ] `DuelingDQNAgent<T>` - Different network architecture
- [ ] `RainbowDQNAgent<T>` - Combines all improvements

### Phase 3: Policy Gradient (Week 3)
- [ ] `REINFORCEAgent<T>`
- [ ] `A2CAgent<T>`
- [ ] `A3CAgent<T>`
- [ ] `PPOAgent<T>`
- [ ] `TRPOAgent<T>`

### Phase 4: Actor-Critic (Week 4)
- [ ] `DDPGAgent<T>`
- [ ] `TD3Agent<T>`
- [ ] `SACAgent<T>`

### Phase 5: Advanced (Weeks 5-6)
- [ ] Model-based: Dreamer, MuZero, World Models
- [ ] Multi-agent: MADDPG, QMIX
- [ ] Offline: CQL, IQL, Decision Transformer

## Example: Complete DQN Implementation

```csharp
public class DQNAgent<T> : IRLAgent<T>, IFullModel<T, Tensor<T>, Vector<T>>
{
    private readonly DeepQNetwork<T> _qNetwork;
    private readonly DeepQNetwork<T> _targetNetwork;
    private readonly UniformReplayBuffer<T> _replayBuffer;
    private readonly INumericOperations<T> _numOps;
    private readonly DQNOptions<T> _options;
    private int _steps;

    public DQNAgent(NeuralNetworkArchitecture<T> architecture, DQNOptions<T> options)
    {
        _options = options;
        _numOps = NumericOperations<T>.Instance;
        _qNetwork = new DeepQNetwork<T>(architecture);
        _targetNetwork = new DeepQNetwork<T>(architecture);
        _replayBuffer = new UniformReplayBuffer<T>(options.BufferSize);
        _steps = 0;
    }

    public Vector<T> SelectAction(Tensor<T> state)
    {
        // Epsilon-greedy
        if (_random.NextDouble() < _numOps.ToDouble(_options.Epsilon))
        {
            // Random action
            return Vector<T>.Random(_options.ActionDim);
        }

        // Greedy action
        var qValues = _qNetwork.Predict(state);
        int bestAction = qValues.ArgMax();
        var action = new Vector<T>(_options.ActionDim);
        action[bestAction] = _numOps.One;
        return action;
    }

    public void StoreExperience(Tensor<T> state, Vector<T> action, T reward, Tensor<T> nextState, bool done)
    {
        _replayBuffer.Add(new Experience<T>(state, action, reward, nextState, done));
    }

    public void Train()
    {
        if (!_replayBuffer.CanSample(_options.BatchSize))
            return;

        var batch = _replayBuffer.Sample(_options.BatchSize);

        // Compute target Q-values
        var targets = new List<Tensor<T>>();
        foreach (var exp in batch)
        {
            var target = exp.Done
                ? exp.Reward
                : _numOps.Add(exp.Reward,
                    _numOps.Multiply(_options.Gamma,
                        _targetNetwork.Predict(exp.NextState).Max()));
            targets.Add(target);
        }

        // Update Q-network
        _qNetwork.Train(batchStates, batchTargets);

        // Update target network periodically
        if (++_steps % _options.TargetUpdateFreq == 0)
        {
            _targetNetwork.CopyParametersFrom(_qNetwork);
        }
    }

    // IFullModel implementation
    public Vector<T> Predict(Tensor<T> input) => SelectAction(input);
    public ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { ModelType = ModelType.DQNAgent };
    public void Train(Tensor<T> input, Vector<T> output) => throw new NotSupportedException();

    // Serialization
    public byte[] Serialize() => /* serialize networks + buffer */;
    public void Deserialize(byte[] data) => /* deserialize */;

    // Other IFullModel members...
}
```

## Benefits of This Approach

1. ✅ **Single Entry Point**: Everything through PredictionModelBuilder
2. ✅ **Consistent API**: Same pattern as supervised/meta-learning
3. ✅ **Reuses Infrastructure**: Optimizers, serialization, metrics, etc.
4. ✅ **Type Safety**: Uses Vector/Matrix/Tensor throughout
5. ✅ **Extensible**: Easy to add new algorithms
6. ✅ **Discoverable**: Users find RL same way they find other models

## Migration from Supervised Learning

Users familiar with supervised learning can easily use RL:

```csharp
// Supervised
var result = await new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new LinearRegression<double>())
    .BuildAsync(xData, yData);

// RL - same pattern!
var result = await new PredictionModelBuilder<double, Tensor<double>, Vector<double>>()
    .ConfigureEnvironment(new CartPoleEnvironment<double>())
    .ConfigureModel(new PPOAgent<double>())
    .BuildAsync(episodes: 1000);
```

The parallel is clear and follows existing patterns.
